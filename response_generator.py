import os
from typing import List, Dict, Any, Optional
import openai
from similarity_search import load_index, load_embedding_model, find_similar_chunks, beautify_text
import pandas as pd


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    
    pass


class SyntheticDataGenerator:
    """
    A class to generate synthetic data based on data descriptions
    found in a research paper, using an LLM.
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model_name: str = "google/gemini-2.0-flash-exp:free",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 index_path: str = "faiss_index.index",
                 text_chunks_path: str = "text_chunks.pkl",
                 max_context_length: int = 8000):
        """
        Initializes the SyntheticDataGenerator.
        
        Args:
            openai_api_key: OpenRouter API key.
            model_name: The LLM to use for generation.
            embedding_model_name: The model for creating text embeddings.
            index_path: Path to the FAISS index of the paper's text.
            text_chunks_path: Path to the pickled text chunks from the paper.
            max_context_length: Maximum context length for the LLM prompt.
        """
        # Set up OpenRouter API key
        if openai_api_key:
            self.api_key = openai_api_key
        elif os.getenv("OPENROUTER_API_KEY"):
            self.api_key = os.getenv("OPENROUTER_API_KEY")
        elif os.getenv("OPENAI_API_KEY"):  # Fallback for compatibility
            self.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenRouter API key must be provided either as parameter or OPENROUTER_API_KEY environment variable")
        
        # Use OpenRouter API endpoint
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model_name = model_name
        self.max_context_length = max_context_length
        
        # Load embedding model and FAISS index
        print("Loading embedding model and FAISS index...")
        self.embedding_model = load_embedding_model(embedding_model_name)
        self.index, self.text_chunks = load_index(index_path, text_chunks_path)
        
        if self.index is None or self.text_chunks is None:
            raise ValueError("Failed to load FAISS index or text chunks.")
        
        print(f"Successfully loaded index with {len(self.text_chunks)} text chunks.")
    
    def get_relevant_context(self, query: str, k: int = 10) -> str:
        """
        Retrieves relevant text chunks from the paper to form a context.
        
        Args:
            query: The query to find relevant text for.
            k: The number of chunks to retrieve.
            
        Returns:
            A string containing the formatted context.
        """
        relevant_chunks = find_similar_chunks(
            query, self.embedding_model, self.index, self.text_chunks, k=k
        )
        
        context_parts = []
        total_length = 0
        
        for i, chunk in enumerate(relevant_chunks):
            chunk_text = beautify_text(chunk)
            page_number = chunk.get("page", "N/A")
            
            formatted_chunk = f"[Chunk {i+1}, Page {page_number}]: {chunk_text}"
            
            if total_length + len(formatted_chunk) > self.max_context_length:
                break
                
            context_parts.append(formatted_chunk)
            total_length += len(formatted_chunk)
        
        return "\n\n".join(context_parts)
    
    def generate_synthetic_data(self, 
                                k: int = 10, 
                                temperature: float = 0.7,
                                max_tokens: int = 4000) -> Dict[str, Any]:
        """
        Generates synthetic data by prompting the LLM with context from the paper.
        
        Args:
            k: The number of context chunks to use.
            temperature: The temperature for the LLM generation.
            max_tokens: The maximum number of tokens for the LLM response.
            
        Returns:
            A dictionary containing the generated data and metadata.
        """
        # A fixed query to find the data description sections of the paper
        context_query = "Detailed description of dataset, features, and data analysis methodology."
        context_string = self.get_relevant_context(context_query, k)
        
        system_prompt = """
You are a helpful AI assistant that extracts the data part of the paper.
You are given a paper that uses data to claim findings.
Your job is to answer what features were used, what these features mean
and generate a sample dataset of 100 records of synthetic data for these features.
You have to keep the synthetic data with the similar distribution in all features.
The collinearity between the synthetic data should remain similar to what is mentioned in the paper.
The distribution of the categorical data should be consistent in the data produced.
The distribution of numerical data should be consistent with either uniform distribution or binomial distribution or normal distribution.
This should be applied to each and every feature.
The output should be only the synthetic data in CSV format, without any explanations or additional text.
"""
        
        user_prompt = f"""Based on the following context from a research paper, please generate a synthetic dataset of 100 records.

Context from the paper:
{context_string}
"""
        
        try:
            # Generate response using OpenRouter
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            generated_text = response.choices[0].message.content
            
            return {
                "response": generated_text,
                "context_used": context_string,
                "model_used": self.model_name,
            }
            
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "context_used": context_string,
                "error": str(e)
            }
    
def main():
    """
    Main function to generate and save synthetic data.
    """
    try:
        # Initialize the generator
        generator = SyntheticDataGenerator(
            model_name="google/gemini-flash-1.5",
            max_context_length=8000
        )
        
        print("Generator initialized. Generating synthetic data...")
        
        # Generate data
        result = generator.generate_synthetic_data(k=10)
        
        if "error" in result:
            print(f"An error occurred: {result['error']}")
        else:
            # Save the synthetic data to a CSV file
            csv_data = result['response']
            try:
                # The LLM is prompted to return only CSV data
                with open("synthetic_data.csv", "w") as f:
                    f.write(csv_data)
                print("Synthetic data saved to synthetic_data.csv")
                
                # Optional: Load into pandas to verify and display
                df = pd.read_csv("synthetic_data.csv")
                print("\nFirst 5 rows of the synthetic dataset:")
                print(df.head())
                
            except Exception as e:
                print(f"Could not process the response as CSV: {e}")
                print("Here is the raw response:")
                print(csv_data)

    except Exception as e:
        print(f"Error initializing or running the generator: {e}")
        print("Please ensure your environment is set up correctly.")


if __name__ == "__main__":
    main()
