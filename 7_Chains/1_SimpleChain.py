from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

template = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

parser= StrOutputParser()

chain = template | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

#chain.get_graph().print_ascii()
