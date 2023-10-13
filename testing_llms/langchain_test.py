import os
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

openai.api_key = os.environ["OPENAI_API_KEY"]

llm = OpenAI(model_name="text-ada-001")

template = """
Interprete the text and evaluate the text.
sentiment: is the text in a positive, neutral or negative sentiment?
subject: What subject is the text about? Use exactly one word.

Just return the JSON, do not add ANYTHING, NO INTERPRETATION!

text: {input}

{format_instructions}

"""
sentiment_schema = ResponseSchema(
    name="sentiment",
    description="Is the text positive, neutral or negative? Only provide these words",
)
subject_schema = ResponseSchema(
    name="subject", description="What subject is the text about? Use exactly one word."
)
price_schema = ResponseSchema(
    name="price",
    description="How expensive was the product? Use None if no price was provided in the text",
)

response_schemas = [sentiment_schema, subject_schema, price_schema]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template(template=template)
messages = prompt.format_messages(
    input="I ordered Pizza Salami for 9.99$ and it was awesome!",
    format_instructions=format_instructions,
)

chat = ChatOpenAI(temperature=0.0)
response = chat(messages)
output_dict = parser.parse(response.content)
print(type(output_dict))
print(output_dict)