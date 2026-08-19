import pymupdf  #  Reads PDFs and extracts text page-by-page
import chromadb #  In-memory vector database storage
from chromadb.utils import embedding_functions # Turns text into embeddings
from google import genai # Google GenAI SDK for Gemini


GEMINI_API_KEY = "APIKEY"
PDF_FILENAME = "pdfname" # Make sure this file exists in your folder


gemini_client = genai.Client(api_key=GEMINI_API_KEY)



# SUB-PART 1: PDF EXTRACTION & CHUNKING MODULE

def extract_and_chunk_pdf(pdf_path, chunk_size=500, overlap=50):
    """
    Opens a PDF using PyMuPDF, reads text page-by-page, strips empty pages,
    and slices text blocks into overlapping chunks while preserving page numbers.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f" Error opening PDF file: {e}")
        return []

    chunks_with_metadata = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        
        # Skip blank/image-only pages with no extractable text layer
        if not text:
            continue
            
        # Slice page string into overlapping blocks so text isn't cut mid-thought
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunks_with_metadata.append({
                "text": chunk_text,
                "page": page_num + 1  # 1-indexed page numbers 
            })
            
            start += (chunk_size - overlap)

    return chunks_with_metadata



# SUB-PART 2: VECTOR EMBEDDING & RAM STORAGE MODULE

def setup_vector_database(chunks):
    """
    Initializes a transient, in-memory ChromaDB instance. It uses the 
    sentence-transformers model to convert chunks into 384-dim vector arrays
    and stores them temporarily in RAM.
    """
    # 1. Initialize temporary RAM client (wipes completely when script terminates)
    chroma_client = chromadb.Client()
    
    # 2. Setup local embedding model (converts strings into math vector arrays)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # 3. Create an in-memory collection container
    collection = chroma_client.create_collection(
        name="session_pdf_memory",
        embedding_function=embedding_fn
    )
    
    # 4. Format chunk lists into arrays for database ingestion
    documents = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"]} for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # 5. Insert documents (Chroma handles embedding generation automatically behind the scenes)
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection



# SUB-PART 3: RETRIEVAL MODULE (SEMANTIC SEARCH)

def retrieve_relevant_chunks(collection, user_query, top_k=3):
    """
    Takes a user question, vectorizes it, and queries the RAM database using 
    similarity matching to return the top K most relevant text chunks and their pages.
    """
    results = collection.query(
        query_texts=[user_query],
        n_results=top_k
    )
    
    return results



# SUB-PART 4: GENERATIVE AUGMENTATION & LLM MODULE

def generate_rag_answer(user_query, search_results):
    """
    Aggregates retrieved chunks into context text, constructs a strict system prompt 
    enforcing page citations, and sends it to the Gemini API for final synthesis.
    """
    # 1. Compile retrieved snippets into an organized context block
    context_block = ""
    for idx in range(len(search_results['documents'][0])):
        text = search_results['documents'][0][idx]
        page = search_results['metadatas'][0][idx]['page']
        context_block += f"--- Source Page {page} ---\n{text}\n\n"

    # 2. Build strict instructions preventing hallucination and forcing citations
    prompt = f"""
    You are an expert document assistant. Answer the user's question using ONLY the provided context below.
    If the answer cannot be found within the context, explicitly state: "I cannot answer this based on the provided document."
    Always cite the exact page number(s) where you found the information.

    DOCUMENT CONTEXT:
    {context_block}

    USER QUESTION: 
    {user_query}
    """

    # 3. Request generation from Google Gemini API
    response = gemini_client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
    )
    
    return response.text



# MAIN APPLICATION CONTROLLER LOOP

def main():
    print(f"\n[1/3] Parsing and chunking PDF: {PDF_FILENAME}...")
    chunks = extract_and_chunk_pdf(PDF_FILENAME)
    
    if not chunks:
        print("Aborting: No chunks were generated. Check your PDF file.")
        return

    print(f"[2/3] Loading {len(chunks)} chunks into temporary RAM database...")
    db_collection = setup_vector_database(chunks)
    
    print("[3/3] Session initialized! RAM is locked and active.\n")
    print("=" * 60)
    print("🤖 MINI-NOTEBOOKLM ACTIVE (Type 'quit' or 'exit' to clear RAM)")
    print("=" * 60)

    # Infinite chat loop keeps the Python process alive and data resident in RAM
    while True:
        user_query = input("\nAsk a question about your PDF: ")
        
        if user_query.lower() in ['quit', 'exit']:
            print("Shutting down session... RAM cleared instantly.")
            break
            
        if not user_query.strip():
            continue

        print("\n Searching vector space...")
        retrieved_data = retrieve_relevant_chunks(db_collection, user_query, top_k=3)

        print("Synthesizing response with Gemini...")
        final_answer = generate_rag_answer(user_query, retrieved_data)

        print("\n" + "~" * 10 + " ANSWER " + "~" * 10)
        print(final_answer)
        print("~" * 28)

if __name__ == "__main__":
    main()