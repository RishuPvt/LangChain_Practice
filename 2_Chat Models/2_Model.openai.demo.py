from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model =ChatOpenAI(model="" , temperature=0.8 , max_completion_tokens=10)
result = model.invoke("what is capital of india")

print(result.content)


#temperature is a parameter that control randomness of lang model outputs . it affects how creative and deterministic the response are.