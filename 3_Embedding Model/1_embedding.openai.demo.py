from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="" , dimensions=32)


result=embedding.embed_query("what is capital of india")

print(result)


#dimensions is how many vector o/p we are expecting .