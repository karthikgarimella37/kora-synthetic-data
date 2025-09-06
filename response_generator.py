import os
from typing import List, Dict, Any, Optional
import openai
from similarity_search import load_index, load_embedding_model, find_similar_chunks, beautify_text

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading .env file
    pass


class ResponseGenerator:
    """
    A response generator that uses similarity search to find relevant context
    and generates responses using OpenRouter API (supporting various LLM models).
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model_name: str = "google/gemini-2.0-flash-exp:free",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 index_path: str = "faiss_index.index",
                 text_chunks_path: str = "text_chunks.pkl",
                 max_context_length: int = 4000):
        """
        Initialize the response generator using OpenRouter API.
        
        Args:
            openai_api_key: OpenRouter API key (if None, will look for OPENROUTER_API_KEY env var)
            model_name: Model to use for response generation (OpenRouter format, e.g., "openai/gpt-3.5-turbo")
            embedding_model_name: HuggingFace embedding model name
            index_path: Path to the FAISS index file
            text_chunks_path: Path to the text chunks pickle file
            max_context_length: Maximum length of context to include in prompt
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
        
        # Load embedding model and index
        print("Loading embedding model and FAISS index...")
        self.embedding_model = load_embedding_model(embedding_model_name)
        self.index, self.text_chunks = load_index(index_path, text_chunks_path)
        
        if self.index is None or self.text_chunks is None:
            raise ValueError("Failed to load index or text chunks. Please ensure the files exist.")
        
        print(f"Successfully loaded index with {len(self.text_chunks)} text chunks")
    
    def get_relevant_context(self, query: str, k: int = 5) -> tuple[List[Dict[str, Any]], str]:
        """
        Find relevant context chunks for the given query.
        
        Args:
            query: The user's query
            k: Number of similar chunks to retrieve
            
        Returns:
            tuple: (list of relevant chunks, formatted context string)
        """
        relevant_chunks = find_similar_chunks(
            query, self.embedding_model, self.index, self.text_chunks, k=k
        )
        
        # Format context for the prompt
        context_parts = []
        total_length = 0
        
        for i, chunk in enumerate(relevant_chunks):
            chunk_text = beautify_text(chunk)
            page_number = chunk.get("page", "N/A")
            
            # Create a formatted chunk with metadata
            formatted_chunk = f"[Chunk {i+1}, Page {page_number}]: {chunk_text}"
            
            # Check if adding this chunk would exceed max context length
            if total_length + len(formatted_chunk) > self.max_context_length:
                break
                
            context_parts.append(formatted_chunk)
            total_length += len(formatted_chunk)
        
        context_string = "\n\n".join(context_parts)
        return relevant_chunks, context_string
    
    def generate_response(self, 
                         query: str, 
                         k: int = 5, 
                         temperature: float = 0.7,
                         max_tokens: int = 500,
                         system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response to the user's query using relevant context.
        
        Args:
            query: The user's query
            k: Number of similar chunks to retrieve for context
            temperature: Sampling temperature for response generation
            max_tokens: Maximum tokens in the generated response
            system_prompt: Custom system prompt (if None, uses default)
            
        Returns:
            dict: Contains 'response', 'relevant_chunks', 'context_used'
        """
        # Get relevant context
        relevant_chunks, context_string = self.get_relevant_context(query, k)
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = """
            You are a helpful AI assistant that extracts the data part of the paper.
            You are given a paper that uses data to claim findings.
            Your job is to answer what features were used, what these features mean
            and generate a sample dataset of 100 records of synthetic data for these features.
            """
        
        # Construct the prompt
        user_prompt = f"""Context information:
{context_string}

Question: {query}

Please provide a comprehensive answer based on the context above. If you reference specific information, 
mention which chunk/page it comes from."""
        
        try:
            # Generate response using OpenAI
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
                "relevant_chunks": relevant_chunks,
                "context_used": context_string,
                "query": query,
                "model_used": self.model_name,
                "chunks_retrieved": len(relevant_chunks)
            }
            
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "relevant_chunks": relevant_chunks,
                "context_used": context_string,
                "query": query,
                "error": str(e)
            }
    
def main():
    """
    Main function to demonstrate the response generator.
    """
    try:
        # Initialize the response generator
        generator = ResponseGenerator(
            model_name="google/gemini-2.0-flash-exp:free",  
            max_context_length=4000
        )
        
        # Example queries
        example_queries = [
            "What are the data features used in the paper and list all the features names."
        ]
        
        print("Response Generator initialized successfully!")
        print("\nTesting with example queries:")
        print("=" * 50)
        
        for query in example_queries:
            print(f"\nQuery: {query}")
            result = generator.generate_response(query, k=3)
            print(f"Response: {result['response']}")
            print(f"Used {result['chunks_retrieved']} chunks")
            print("-" * 30)
        
    except Exception as e:
        print(f"Error initializing response generator: {e}")
        print("Please ensure:")
        print("1. OPENROUTER_API_KEY environment variable is set")
        print("2. faiss_index.index and text_chunks.pkl files exist")
        print("3. All required dependencies are installed")
        print("4. Get your OpenRouter API key from: https://openrouter.ai/")


if __name__ == "__main__":
    main()
