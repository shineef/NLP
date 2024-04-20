from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pprint import pprint
# import json
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model_name="claude-3-opus-20240229", anthropic_api_key="")
llm = ChatOpenAI(openai_api_key = "sk-")

first_prompt = PromptTemplate.from_template("translate this review to English: {review}")
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key = 'English_review')
second_prompt = PromptTemplate.from_template("please summarize this review in one sentence: {English_review}")
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key = 'summary')
third_prompt=ChatPromptTemplate.from_template("Which language this review belongs to?: {review}")
chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key = 'language')
fourth_prompt =ChatPromptTemplate.from_template("Use the required language to write the followup message for the input comment."
                                                "\n summary:{summary}\n language:{language}")
chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key = 'followup_message')
five_prompt =ChatPromptTemplate.from_template("Translate the input message to English. {followup_message}")
chain_five = LLMChain(llm=llm, prompt=five_prompt, output_key = 'English_followup_message')
overall_simple_chain=SequentialChain(
    chains = [chain_one, chain_two, chain_three, chain_four, chain_five],
    input_variables =['review'],
    output_variables = ['language', 'English_review', 'summary', 'followup_message', 'English_followup_message'],
    verbose=True)

review="Je trouve que la glace dans cette crèmerie est assez ordinaire. \
    On sent un fort godt d'arómes artificiels, certaines glaces sont trop acides, \
    voire pas très fraiches, Est-ce gue le personnel ici triche sur la qualité et ne  \
    prend pas soin de faire des produits de qualité ?"

result = overall_simple_chain(review)

# print(json.dumps(result, indent=2))
pprint(result)