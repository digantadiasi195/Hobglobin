#rag_pipeline.py
import os
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.docstore.document import Document
import google.generativeai as genai
from dotenv import load_dotenv

class RAGPipeline:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        # Use HuggingFaceEmbeddings from langchain-huggingface
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = self._build_vector_store()

    def _extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    def _build_vector_store(self):
        data_folder = "data"
        documents = []
        for filename in os.listdir(data_folder):
            if filename.endswith(".pdf"):
                text = self._extract_text_from_pdf(os.path.join(data_folder, filename))
                documents.append(Document(page_content=text, metadata={"source": filename}))
        
        # Pass the HuggingFaceEmbeddings instance to FAISS
        return FAISS.from_documents(documents, self.embedding_model)

    def generate_fine_prints(self):
        context = "\n".join([doc.page_content for doc in self.vector_store.docstore._dict.values()])
        prompt = f"""
        From the following project documents, extract key details ('fine-prints') critical for drafting project proposals.
        Focus on mandatory documents, permits, approvals, security requirements, and site access protocols.
        Summarize the details concisely in bullet points:
        {context}
        """
        response = self.model.generate_content(prompt)
        return response.text

    def chat(self, query):
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(query, k=2)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"""
        Using the following context, answer the query concisely and accurately:
        Context: {context}
        Query: {query}
        """
        response = self.model.generate_content(prompt)
        return response.text