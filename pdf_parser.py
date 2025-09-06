import os
import tempfile
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core.node_parser import SimpleNodeParser
from dotenv import load_dotenv

def pdf_parser(pdf_file):
    load_dotenv()
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY is not set in the environment variables.")

    pdf_content = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name

    nest_asyncio.apply()

    parser = LlamaParse(api_key=api_key, result_type="text", verbose=True)
    documents = parser.load_data(temp_file_path)
    
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
    nodes = node_parser.get_nodes_from_documents(documents)

    for node in nodes:
        pdf_content.append({
            'text': node.get_content(),
            'page': node.metadata.get('page_label', 'N/A')
        })

    os.unlink(temp_file_path)
    print(f"Parsed {len(pdf_content)} chunks from the PDF.")
    print(f"{pdf_content}")
    return pdf_content


if __name__ == "__main__":
    pdf_parser(open("FP-Juliett-Final-Report.pdf", "rb"))
    