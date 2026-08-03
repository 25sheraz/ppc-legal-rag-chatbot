import os
import json
import glob
import numpy as np
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

try:
    from rapidocr_onnxruntime import RapidOCR
    ocr_engine = RapidOCR()
except Exception:
    ocr_engine = None

load_dotenv()

INDEX_DIR = os.path.join(os.path.dirname(__file__), "vector_store")
EMBEDDINGS_FILE = os.path.join(INDEX_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.json")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set in .env file.")

client = OpenAI(api_key=api_key)

def ocr_page_image(page, dpi=150):
    """Fallback OCR for scanned PDF pages using PyMuPDF pixmap & RapidOCR"""
    if ocr_engine is None:
        return ""
    try:
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        result, _ = ocr_engine(img_bytes)
        if result:
            lines = [line[1] for line in result if line[1] and line[1].strip()]
            return "\n".join(lines)
    except Exception:
        pass
    return ""

def extract_chunks_from_pdf(pdf_path, chunk_size=800, chunk_overlap=120):
    """
    Extracts text page-by-page from a given PDF document.
    """
    doc_name = os.path.basename(pdf_path)
    print(f"\nProcessing PDF: '{doc_name}' via PyMuPDF...")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening '{pdf_path}': {e}")
        return []

    total_pages = len(doc)
    print(f"Total pages in '{doc_name}': {total_pages}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]
        
        # 1. Try direct text extraction
        page_text = page.get_text()
        
        # Clean non-breaking spaces
        if page_text:
            page_text = page_text.replace("\xa0", " ")

        # 2. Perform OCR if direct text is missing or extremely short
        if not page_text or len(page_text.strip()) < 40:
            ocr_text = ocr_page_image(page)
            if ocr_text and len(ocr_text.strip()) > 30:
                page_text = ocr_text

        if not page_text or not page_text.strip():
            continue

        page_chunks = text_splitter.split_text(page_text.strip())
        for text_chunk in page_chunks:
            if text_chunk.strip():
                chunks.append({
                    "document": doc_name,
                    "page": page_num,
                    "text": text_chunk.strip()
                })

    print(f"--> Extracted {len(chunks)} text chunks from '{doc_name}'")
    return chunks

def build_index(pdf_files=None, force_reindex=False):
    """
    Finds all target PDF books (e.g. 'pakistan penal code.pdf'),
    extracts chunks, computes OpenAI embeddings, and saves local vector store.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)

    if pdf_files is None:
        workspace_dir = os.path.dirname(__file__)
        pdf_files = [
            f for f in glob.glob(os.path.join(workspace_dir, "*.pdf"))
            if not f.endswith("_clean.pdf") and not f.endswith("_fixed.pdf") and not f.endswith("_original.pdf") and not f.endswith("_repaired.pdf")
        ]

    print(f"Target PDF books to index: {[os.path.basename(p) for p in pdf_files]}")

    all_chunks = []
    chunk_id = 0
    for pdf_path in pdf_files:
        doc_chunks = extract_chunks_from_pdf(pdf_path)
        for c in doc_chunks:
            c["id"] = chunk_id
            all_chunks.append(c)
            chunk_id += 1

    if not all_chunks:
        print("No valid text chunks extracted from PDFs!")
        return

    print(f"\nTotal combined chunks from all books: {len(all_chunks)}")
    print("Generating embeddings via OpenAI text-embedding-3-small...")

    batch_size = 50
    all_embeddings = []

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        texts = [item["text"] for item in batch]

        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            print(f"Embedded chunks {i+1} to {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")
        except Exception as e:
            print(f"Error during embedding batch {i}: {e}")
            raise e

    embeddings_matrix = np.array(all_embeddings, dtype=np.float32)

    np.save(EMBEDDINGS_FILE, embeddings_matrix)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Successfully saved multi-book index to '{INDEX_DIR}'")
    print(f"Embeddings shape: {embeddings_matrix.shape}")
    print(f"Total indexed chunks: {len(all_chunks)}")

if __name__ == "__main__":
    build_index(force_reindex=True)
