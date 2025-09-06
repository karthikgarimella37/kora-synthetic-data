#!/usr/bin/env python3
"""
Simple script to test if OpenRouter API key is properly loaded from environment.
"""

import os

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ dotenv loaded successfully")
except ImportError:
    print("⚠️  python-dotenv not available - install with: pip install python-dotenv")

# Check for OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")  # Fallback compatibility

if api_key:
    # Show only first 8 and last 4 characters for security
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    key_source = "OPENROUTER_API_KEY" if os.getenv("OPENROUTER_API_KEY") else "OPENAI_API_KEY"
    print(f"✅ API key found from {key_source}: {masked_key}")
    
    # Test basic OpenRouter connection
    try:
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Make a simple test call using a free model
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",  # OpenRouter format
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print("✅ OpenRouter API connection successful!")
        print(f"Test response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ OpenRouter API connection failed: {e}")
        print("Make sure your OpenRouter API key is valid and has credits/usage available")
        
else:
    print("❌ OpenRouter API key not found in environment variables")
    print("\nTo fix this, make sure you have a .env file with:")
    print("OPENROUTER_API_KEY=your-openrouter-api-key-here")
    print("\nOr set it directly in your shell:")
    print("export OPENROUTER_API_KEY=your-openrouter-api-key-here")
    print("\nGet your free OpenRouter API key from: https://openrouter.ai/")

# Check for required files
print("\nChecking required files:")
required_files = ["faiss_index.index", "text_chunks.pkl"]

for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file} found")
    else:
        print(f"❌ {file} missing - run embedding generation first")

print("\nIf all checks pass, you should be able to use the response generator!")
