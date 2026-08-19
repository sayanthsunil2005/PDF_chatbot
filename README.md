# PDF_chatbot
this is a chat bot in which u can upload PDF and can ask things about the contents in the PDF

Additional packages needed


pymupdf: Used to open the PDF, read the text layer, and track the exact page numbers for citations.

chromadb: The core vector database that runs in RAM. It stores the text chunks and performs the mathematical similarity search (nearest-neighbor).

sentence-transformers: Provides the all-MiniLM-L6-v2 embedding model that ChromaDB uses to translate English text into 384-dimensional math vectors.

google-genai: The official Google SDK used to connect to the gemini-3.6-flash model, which reads the retrieved chunks and writes the final conversational answer.

commad to donlod these packages : "pip install pymupdf chromadb sentence-transformers google-genai"
