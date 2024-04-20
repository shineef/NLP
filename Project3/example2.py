from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pprint import pprint

llm = ChatOpenAI(openai_api_key="sk-")

# subjects list
subjects = ["Math", "History", "Computer Science"]

# if prompt related to subject list
def is_subject_related(input_text):
    prompt = f"Is the text: {input_text} related to the following subjects: {subjects}? The answer is only yes or no."
    answer_output = llm.invoke(prompt)
    if "yes" in answer_output.content.lower():
        return True
    else:
        return False

# destination chain, process subject related input
def destination_chain(input_text, subject):
    prompt = ChatPromptTemplate.from_template(f"Please answer the following text: {input_text} with a very detailed explanation on {subject}")
    chain = LLMChain(llm=llm, prompt=prompt, output_key=f'output')
    return chain

# default chain, process non-subject related input
def default_chain(input_text):
    prompt = ChatPromptTemplate.from_template(f"Just answer the following text casually: {input_text}")
    chain = LLMChain(llm=llm, prompt=prompt, output_key='output')
    return chain

def process_input(input_text):
    subject = is_subject_related(input_text)
    if subject:
        print("Subject related")
        return destination_chain(input_text, subject)
    else:
        print("Not subject related")
        return default_chain(input_text)

# input_text = "Please explain the Mona Lisa"
input_text = "Please explain the Pythagoras theorem"

subject_chain = process_input(input_text)
prompt1 = ChatPromptTemplate.from_template("translate this text into Chinese: {output}")
translate_chain = LLMChain(llm=llm, prompt=prompt1, output_key = 'translated_output')
prompt2 = ChatPromptTemplate.from_template("summary this text in one sentence: {output}")
summary_chain = LLMChain(llm=llm, prompt=prompt2, output_key = 'summarized_output')
overall_simple_chain=SequentialChain(
    chains = [subject_chain, translate_chain, summary_chain],
    input_variables =['input_text'],
    output_variables = ['output', 'translated_output', 'summarized_output'],
    verbose=True)
result = overall_simple_chain(input_text)
pprint(result)
