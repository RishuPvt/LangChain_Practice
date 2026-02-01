from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)
class City(BaseModel):
    fact:str = Field(description='Intresting Fact about the {topic}')
    Economy:int = Field(description='Economy Growth Rate of the {topic}')
    Country: str = Field(description='Name of the Country that {topic} belongs to')


parser= PydanticOutputParser(pydantic_object=City)

template =PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


chain = template | model | parser

result = chain.invoke({'topic':'delhi'})

print(result)