from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

gemini = GoogleGenerativeAI(  model="gemini-3-flash-preview")

result = gemini.invoke("what is capital of india")

print(result)