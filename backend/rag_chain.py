import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from supabase.client import create_client, Client

# Set your keys (hardcoded as provided; move to env for security)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAm_B7ME_BVTE89Wc8Lg-FLsBjxXP-b1Q0"
SUPABASE_URL = "https://rfckjjhbbvuirbtczvod.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJmY2tqamhiYnZ1aXJidGN6dm9kIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ5MjQ2ODAsImV4cCI6MjA3MDUwMDY4MH0.MJtrn-eJHLdRaijUbc8ZLAlJW6U0AFtZNorpbWrtb0E"

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def build_rag_chain():
    # Load company data from file
    loader = TextLoader("data/company_data.txt")
    documents = loader.load()
    
    # Split into chunks for better retrieval
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store (Supabase cloud)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Gemini embeddings
    
    # Upload to Supabase (adds if not exist; run once or when data changes)
    vector_store = SupabaseVectorStore.from_documents(
        texts,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",  # Matches the SQL function
        chunk_size=500  # Batch upload
    )
    
    # LLM model (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    
    # RAG chain: Retrieve + Generate
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Stuff all retrieved docs into prompt
        retriever=vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant chunks
    )
    return qa_chain

# Global chain (build once)
rag_chain = build_rag_chain()