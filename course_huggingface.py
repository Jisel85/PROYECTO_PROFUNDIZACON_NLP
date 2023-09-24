import logging
import time
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================================================================= #
# ====================================== HuggingFace ========================================================== #
# ============================================================================================================= #
sentiment_classifier = pipeline("sentiment-analysis")
print(sentiment_classifier("Colombia es un país muy feo, pero me gusta vivir en él."))
print(
    sentiment_classifier(
        [
            "I've been waiting for a HuggingFace course my whole life.",
            "I hate this so much!",
        ]
    )
)

label_classifier = pipeline("zero-shot-classification")

print(
    label_classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )
)

generator = pipeline("text-generation")
print(generator("In this course, we will teach you how to"))

generator = pipeline(task="text-generation", model="distilgpt2")
print(
    generator(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,
    )
)

summarizer = pipeline("summarization")
print(
    summarizer(
        """
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers.
"""
    )
)


# ============================================================================================================= #
# ========================================= Langchain ========================================================= #
# ============================================================================================================= #
logging.info(f"New LLM")
logging.info("Start")

llm = HuggingFacePipeline.from_model_id(
    model_id="distilgpt2",
    task="text-generation",
    model_kwargs={"max_length": 64},
)

template_text = """{full_text}"""
prompt = PromptTemplate.from_template(template_text)
chain = prompt | llm
full_text = "In this course, we will teach you how to"
print(f"Input text: {full_text}")
print(f"LLM's answer:\n {chain.invoke({'full_text': full_text})}")
logging.info("End\n")
time.sleep(3)

# ============================================================================================================= #
# ============================================================================================================= #
# ============================================================================================================= #
logging.info(f"New LLM")
logging.info("Start")

llm = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    model_kwargs={
        "max_length": 128
    }
)

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)
chain = prompt | llm
question = "What is electroencephalography?"
print(f"The question is: {question}")
print(f"LLM's answer:\n {chain.invoke({'question': question})}")
logging.info("End\n")
time.sleep(3)

# ============================================================================================================= #
# ============================================================================================================= #
# ============================================================================================================= #
logging.info(f"New LLM")
logging.info("Start")

llm = HuggingFacePipeline.from_model_id(
    model_id="sshleifer/distilbart-cnn-12-6",
    task="summarization",
)
summarization_template = """{full_text}"""
prompt = PromptTemplate.from_template(summarization_template)

full_text = """
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers.
"""
chain = prompt | llm
print(f"FULL TEXT:\n{full_text}")
print(f"LLM's answer:\nSUMMARY:\n")
print(chain.invoke({"full_text": full_text}))
time.sleep(3)
logging.info("End")

# ============================================================================================================= #
# ============================================================================================================= #
# ============================================================================================================= #
logging.info(f"New LLM")
logging.info("Start")

llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    model_kwargs={
        "max_length": 30,
    }
)

text_template = """{text}"""
prompt = PromptTemplate.from_template(text_template)
input_text = "Hello, I'm a language model,"
chain = prompt | llm
print(f"Input: {input_text}")
print(f"LLM's answer:\n")
print(chain.invoke({"text": input_text}))
logging.info("End")
#
#
#
#
#
# import logging
# import time
# from transformers import pipeline
# from langchain.llms import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
#
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
#
#
# # # # ============================================================================================================= #
# # # # ====================================== HuggingFace ========================================================== #
# # # # ============================================================================================================= #
# # sentiment_classifier = pipeline("sentiment-analysis")
# # print(sentiment_classifier("Colombia es un país muy feo, pero me gusta vivir en él."))
# # print(
# #     sentiment_classifier(
# #         [
# #             "I've been waiting for a HuggingFace course my whole life.",
# #             "I hate this so much!",
# #         ]
# #     )
# # )
# #
# # label_classifier = pipeline("zero-shot-classification")
# #
# # print(
# #     label_classifier(
# #         "This is a course about the Transformers library",
# #         candidate_labels=["education", "politics", "business"],
# #     )
# # )
# #
# # generator = pipeline("text-generation")
# # print(generator("In this course, we will teach you how to"))
# #
# # generator = pipeline(task="text-generation", model="distilgpt2")
# # print(
# #     generator(
# #         "In this course, we will teach you how to",
# #         max_length=30,
# #         num_return_sequences=2,
# #     )
# # )
# #
# # summarizer = pipeline("summarization")
# # print(
# #     summarizer(
# #         """
# #     America has changed dramatically during recent years. Not only has the number of
# #     graduates in traditional engineering disciplines such as mechanical, civil,
# #     electrical, chemical, and aeronautical engineering declined, but in most of
# #     the premier American universities engineering curricula now concentrate on
# #     and encourage largely the study of engineering science. As a result, there
# #     are declining offerings in engineering subjects dealing with infrastructure,
# #     the environment, and related issues, and greater concentration on high
# #     technology subjects, largely supporting increasingly complex scientific
# #     developments. While the latter is important, it should not be at the expense
# #     of more traditional engineering.
# #
# #     Rapidly developing economies such as China and India, as well as other
# #     industrial countries in Europe and Asia, continue to encourage and advance
# #     the teaching of engineering. Both China and India, respectively, graduate
# #     six and eight times as many traditional engineers as does the United States.
# #     Other industrial countries at minimum maintain their output, while America
# #     suffers an increasingly serious decline in the number of engineering graduates
# #     and a lack of well-educated engineers.
# # """
# #     )
# # )
#
#
# # # ============================================================================================================= #
# # # ========================================= Langchain ========================================================= #
# # # ============================================================================================================= #
# logging.info(f"\n{'='*100}\nStar use with Langchain\n{'='*100}")
# logging.info(f"New LLM")
# logging.info("Start")
#
# llm = HuggingFacePipeline.from_model_id(
#     model_id="distilgpt2",
#     task="text-generation",
#     model_kwargs={"max_length": 64},
# )
#
# template_text = """{full_text}"""
# prompt = PromptTemplate.from_template(template_text)
# chain = prompt | llm
# full_text = "In this course, we will teach you how to"
# print(full_text)
# print(chain.invoke({"full_text": full_text}))
# logging.info("End\n")
# time.sleep(3)
#
# # ============================================================================================================= #
# # ============================================================================================================= #
# # ============================================================================================================= #
# logging.info(f"New LLM")
# logging.info("Start")
#
# llm = HuggingFacePipeline.from_model_id(
#     model_id="bigscience/bloom-1b7",
#     task="text-generation",
#     model_kwargs={
#         "max_length": 128
#     }
# )
#
# template = """Question: {question}
#
# Answer: Let's think step by step."""
# prompt = PromptTemplate.from_template(template)
# chain = prompt | llm
# question = "What is electroencephalography?"
# print(f"The question is: {question}")
# print(f"LLM's answer:\n: {chain.invoke({'question': question})}")
# logging.info("End\n")
# time.sleep(3)
#
# # ============================================================================================================= #
# # ============================================================================================================= #
# # ============================================================================================================= #
# logging.info(f"New LLM")
# logging.info("Start")
#
# llm = HuggingFacePipeline.from_model_id(
#     model_id="sshleifer/distilbart-cnn-12-6",
#     task="summarization",
# )
# summarization_template = """{full_text}"""
# prompt = PromptTemplate.from_template(summarization_template)
#
# full_text = """
#     America has changed dramatically during recent years. Not only has the number of
#     graduates in traditional engineering disciplines such as mechanical, civil,
#     electrical, chemical, and aeronautical engineering declined, but in most of
#     the premier American universities engineering curricula now concentrate on
#     and encourage largely the study of engineering science. As a result, there
#     are declining offerings in engineering subjects dealing with infrastructure,
#     the environment, and related issues, and greater concentration on high
#     technology subjects, largely supporting increasingly complex scientific
#     developments. While the latter is important, it should not be at the expense
#     of more traditional engineering.
#
#     Rapidly developing economies such as China and India, as well as other
#     industrial countries in Europe and Asia, continue to encourage and advance
#     the teaching of engineering. Both China and India, respectively, graduate
#     six and eight times as many traditional engineers as does the United States.
#     Other industrial countries at minimum maintain their output, while America
#     suffers an increasingly serious decline in the number of engineering graduates
#     and a lack of well-educated engineers.
# """
# chain = prompt | llm
# print(f"FULL TEXT:\n{full_text}")
# print(f"LLM's answer:\nSUMMARY:\n")
# print(chain.invoke({"full_text": full_text}))
# logging.info("End")
# time.sleep(3)
#
# # ============================================================================================================= #
# # ============================================================================================================= #
# # ============================================================================================================= #
# logging.info(f"New LLM")
# logging.info("Start")
#
# llm = HuggingFacePipeline.from_model_id(
#     model_id="gpt2",
#     task="text-generation",
#     model_kwargs={
#         "max_length": 30,
#     }
# )
#
# text_template = """{text}"""
# prompt = PromptTemplate.from_template(text_template)
# input_text = "Hello, I'm a language model,"
# chain = prompt | llm
# print(f"Input: {input_text}")
# print(f"LLM's answer:\n")
# print(chain.invoke({"text": input_text}))
# logging.info("End")
#
#
#
#
#
#
#
