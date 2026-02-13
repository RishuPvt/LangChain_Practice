from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

load_dotenv()

# Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004"
)

# Documents
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history.",
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni is known as Captain Cool and a legendary finisher.",
        metadata={"team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is one of the best fast bowlers in T20 cricket.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder.",
        metadata={"team": "Chennai Super Kings"}
    )
]

# Vector store
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="my_chroma_db",
    collection_name="sample"
)

# Add docs
vector_store.add_documents(docs)

# Similarity search
results = vector_store.similarity_search(
    query="Who is a bowler?",
    k=2
)

print(results)

# Filtered search
filtered = vector_store.similarity_search(
    query="CSK player",
    filter={"team": "Chennai Super Kings"}
)

print(filtered)
