from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
from pdf_parser import pdf_parser
import pickle
import numpy as np


def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    print(f"Loading embedding model: {model_name}")
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    return embed_model


def generate_embedding(pdf_content, embed_model):
    texts = [item['text'] for item in pdf_content]
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress_bar=True)
    return np.array(embeddings, dtype='float32')


def generate_faiss_index(embeddings):
    print('Generating FAISS index...')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index


if __name__ == "__main__":
    pdf_content = pdf_parser(open("FP-Juliett-Final-Report.pdf", "rb"))

    embed_model = load_embedding_model()
    embeddings = generate_embedding(pdf_content, embed_model)
    faiss_index = generate_faiss_index(embeddings)
    
    if faiss_index:
        faiss.write_index(faiss_index, "faiss_index.index")
        print('FAISS index generated successfully and saved to faiss_index.index')

        chunks_file = "text_chunks.pkl"
        with open(chunks_file, "wb") as f:
            pickle.dump(pdf_content, f)

        print(f'Text chunks saved to {chunks_file}')



