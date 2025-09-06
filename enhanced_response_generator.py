import os
from typing import List, Dict, Any, Optional, Union
import openai
from similarity_search import load_index, load_embedding_model, find_similar_chunks, beautify_text

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading .env file
    pass

# Optional imports for alternative LLMs
try:
    from langchain.llms import GPT4All
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    GPT4ALL_AVAILABLE = True
except ImportError:
    GPT4ALL_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class EnhancedResponseGenerator:
    """
    An enhanced response generator that supports multiple LLM backends:
    - OpenAI (GPT-3.5/GPT-4)
    - GPT4All (local models)
    - HuggingFace Transformers (local models)
    """
    
    def __init__(self, 
                 llm_backend: str = "openai",
                 openai_api_key: Optional[str] = None,
                 openai_model: str = "gpt-3.5-turbo",
                 local_model_path: Optional[str] = None,
                 hf_model_name: str = "microsoft/DialoGPT-medium",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 index_path: str = "faiss_index.index",
                 text_chunks_path: str = "text_chunks.pkl",
                 max_context_length: int = 4000,
                 device: str = "auto"):
        """
        Initialize the enhanced response generator.
        
        Args:
            llm_backend: LLM backend to use ("openai", "gpt4all", "huggingface")
            openai_api_key: OpenAI API key
            openai_model: OpenAI model name
            local_model_path: Path to local GPT4All model
            hf_model_name: HuggingFace model name
            embedding_model_name: HuggingFace embedding model name
            index_path: Path to the FAISS index file
            text_chunks_path: Path to the text chunks pickle file
            max_context_length: Maximum length of context to include in prompt
            device: Device for local models ("cpu", "cuda", "auto")
        """
        self.llm_backend = llm_backend.lower()
        self.max_context_length = max_context_length
        
        # Set device for local models
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load embedding model and index
        print("Loading embedding model and FAISS index...")
        self.embedding_model = load_embedding_model(embedding_model_name)
        self.index, self.text_chunks = load_index(index_path, text_chunks_path)
        
        if self.index is None or self.text_chunks is None:
            raise ValueError("Failed to load index or text chunks. Please ensure the files exist.")
        
        print(f"Successfully loaded index with {len(self.text_chunks)} text chunks")
        
        # Initialize the selected LLM backend
        self._initialize_llm_backend(openai_api_key, openai_model, local_model_path, hf_model_name)
    
    def _initialize_llm_backend(self, openai_api_key, openai_model, local_model_path, hf_model_name):
        """Initialize the selected LLM backend."""
        
        if self.llm_backend == "openai":
            self._initialize_openai(openai_api_key, openai_model)
        elif self.llm_backend == "gpt4all":
            self._initialize_gpt4all(local_model_path)
        elif self.llm_backend == "huggingface":
            self._initialize_huggingface(hf_model_name)
        else:
            raise ValueError(f"Unsupported LLM backend: {self.llm_backend}")
    
    def _initialize_openai(self, api_key, model):
        """Initialize OpenAI backend."""
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key required for OpenAI backend")
        
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.openai_model = model
        print(f"Initialized OpenAI backend with model: {model}")
    
    def _initialize_gpt4all(self, model_path):
        """Initialize GPT4All backend."""
        if not GPT4ALL_AVAILABLE:
            raise ImportError("GPT4All not available. Install with: pip install gpt4all")
        
        if not model_path:
            # Default GPT4All model
            model_path = "orca-mini-3b.ggmlv3.q4_0.bin"
        
        try:
            self.gpt4all_model = GPT4All(
                model=model_path,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            print(f"Initialized GPT4All backend with model: {model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load GPT4All model: {e}")
    
    def _initialize_huggingface(self, model_name):
        """Initialize HuggingFace Transformers backend."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available. Install with: pip install transformers torch")
        
        try:
            print(f"Loading HuggingFace model: {model_name} on {self.device}")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Add padding token if not present
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            
            print(f"Initialized HuggingFace backend with model: {model_name}")
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace model: {e}")
    
    def get_relevant_context(self, query: str, k: int = 5) -> tuple[List[Dict[str, Any]], str]:
        """Find relevant context chunks for the given query."""
        relevant_chunks = find_similar_chunks(
            query, self.embedding_model, self.index, self.text_chunks, k=k
        )
        
        # Format context for the prompt
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
        
        context_string = "\n\n".join(context_parts)
        return relevant_chunks, context_string
    
    def _generate_openai_response(self, system_prompt: str, user_prompt: str, 
                                temperature: float, max_tokens: int) -> str:
        """Generate response using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def _generate_gpt4all_response(self, prompt: str, max_tokens: int) -> str:
        """Generate response using GPT4All."""
        full_prompt = f"{prompt}\n\nResponse:"
        response = self.gpt4all_model(full_prompt, max_tokens=max_tokens)
        return response.strip()
    
    def _generate_huggingface_response(self, prompt: str, max_tokens: int, 
                                     temperature: float) -> str:
        """Generate response using HuggingFace Transformers."""
        inputs = self.hf_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        if self.device == "cuda":
            inputs = inputs.cuda()
        
        with torch.no_grad():
            outputs = self.hf_model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.hf_tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        # Decode only the new tokens (response)
        response = self.hf_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def generate_response(self, 
                         query: str, 
                         k: int = 5, 
                         temperature: float = 0.7,
                         max_tokens: int = 500,
                         system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate a response using the selected LLM backend."""
        
        # Get relevant context
        relevant_chunks, context_string = self.get_relevant_context(query, k)
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
Use the context information to provide accurate and relevant answers. If the context doesn't contain 
enough information to answer the question, say so clearly. Always cite the relevant chunks/pages when possible."""
        
        # Construct prompts based on backend
        if self.llm_backend == "openai":
            user_prompt = f"""Context information:
{context_string}

Question: {query}

Please provide a comprehensive answer based on the context above."""
            
            try:
                generated_text = self._generate_openai_response(
                    system_prompt, user_prompt, temperature, max_tokens
                )
            except Exception as e:
                return {"response": f"Error with OpenAI: {str(e)}", "error": str(e)}
        
        else:
            # For local models, combine system and user prompts
            full_prompt = f"""{system_prompt}

Context information:
{context_string}

Question: {query}

Answer:"""
            
            try:
                if self.llm_backend == "gpt4all":
                    generated_text = self._generate_gpt4all_response(full_prompt, max_tokens)
                elif self.llm_backend == "huggingface":
                    generated_text = self._generate_huggingface_response(
                        full_prompt, max_tokens, temperature
                    )
            except Exception as e:
                return {"response": f"Error with {self.llm_backend}: {str(e)}", "error": str(e)}
        
        return {
            "response": generated_text,
            "relevant_chunks": relevant_chunks,
            "context_used": context_string,
            "query": query,
            "model_backend": self.llm_backend,
            "chunks_retrieved": len(relevant_chunks)
        }
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        print(f"Starting interactive chat session with {self.llm_backend} backend.")
        print("Type 'quit' to exit.")
        print("=" * 50)
        
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("\nSearching for relevant information...")
                result = self.generate_response(query)
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                    continue
                
                print(f"\nResponse (using {result.get('chunks_retrieved', 0)} chunks):")
                print("-" * 40)
                print(result["response"])
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function with backend selection."""
    print("Enhanced Response Generator")
    print("Available backends:")
    print("1. OpenAI (requires API key)")
    print("2. GPT4All (local model)")
    print("3. HuggingFace (local model)")
    
    backend_choice = input("\nSelect backend (1/2/3) or enter backend name: ").strip()
    
    backend_map = {"1": "openai", "2": "gpt4all", "3": "huggingface"}
    backend = backend_map.get(backend_choice, backend_choice.lower())
    
    try:
        generator = EnhancedResponseGenerator(llm_backend=backend)
        
        # Test with example query
        test_query = "What is the title of the paper?"
        print(f"\nTesting with query: {test_query}")
        result = generator.generate_response(test_query, k=3)
        
        if "error" not in result:
            print(f"Response: {result['response']}")
            print(f"Used {result['chunks_retrieved']} chunks")
        else:
            print(f"Error: {result['error']}")
        
        # Start interactive session
        generator.interactive_chat()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
