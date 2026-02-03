from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch , RunnableLambda

load_dotenv()
model = ChatGroq(
         model="llama-3.1-8b-instant",
    
)

parser = StrOutputParser()



class Feedback(BaseModel):
        sentiment :Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)


sub_chain = prompt1 | model | parser2
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = sub_chain | branch_chain

result =chain.invoke({'feedback': 'This is a beautiful phone'})
print(sub_chain.invoke({'feedback': 'This is a beautiful phone'}))
print("Response:", result)