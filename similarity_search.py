import faiss
import pickle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np


def load_index(index_path="faiss_index.index", text_chunks_path="text_chunks.pkl"):
    try:
        index = faiss.read_index(index_path)

        with open(text_chunks_path, "rb") as f:
            text_chunks = pickle.load(f)
        return index, text_chunks

    except Exception as e:
        print(f'Error loading index: {e}')


def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    print(f"Loading embedding model: {model_name}")
    model = HuggingFaceEmbedding(model_name=model_name)
    print(f"HuggingFace embedding model {model_name} loaded successfully.")
    return model


def find_similar_chunks(query: str, model, index, text_chunks, k=10):
    query_embedding = model.get_query_embedding(query)
    query_embedding_np = np.array([query_embedding], dtype='float32')
    distances, indices = index.search(query_embedding_np, k=k)

    relevant_chunks = [text_chunks[i] for i in indices[0]]
    return relevant_chunks


def beautify_text(chunk_data):
    text = chunk_data.get("text", "")
    return text.replace('\n', ' ').strip()


if __name__ == "__main__":
    index, text_chunks = load_index()
    model = load_embedding_model()
    query = "What is the title of the paper?"
    relevant_chunks = find_similar_chunks(query, model, index, text_chunks, k=10)
    if relevant_chunks:
        print(f"Printing top {len(relevant_chunks)} similar chunks...")
        for i, chunk in enumerate(relevant_chunks):
            page_number = chunk.get("page", "N/A")
            print(f"\n--Chunk {i+1} (Page: {page_number}) --")
            print(beautify_text(chunk))



