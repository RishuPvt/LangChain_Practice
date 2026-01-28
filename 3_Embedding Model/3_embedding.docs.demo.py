from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

document=[
    "delhi is capital of india"
    "kolkata is capital of west bengal"
    "patna is capital of bihar"
    "jaipur is capital of rajasthan"
]

result = embedding.embed_documents(document)
print(str(result))
