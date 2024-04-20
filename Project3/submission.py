######################## Task 1 ########################

from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import json

# load the pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# read the SQuAD dev dataset
with open('./dev-v2.0.json', 'r', encoding='utf-8') as file:
    squad_data = json.load(file)

# distract ids, questions and contexts
questions_and_contexts = []
for article in squad_data['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            id = qa['id']
            question = qa['question']
            questions_and_contexts.append((id, question, context))

# print(questions_and_contexts[:2])

# conduct predictions
outputs = {}

for id, question, context in questions_and_contexts: 
    # print(f"Question: {question}")
    # print(f"Context: {context}")
    # print(type(question), type(context))
    # inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
    input_ids = tokenizer.encode(question, context)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(len(tokens))

    answer_dict = {}

    if len(input_ids) > 512:
        # Split the input_ids into chunks of 512 tokens
        chunks = [input_ids[i:i + 512] for i in range(0, len(input_ids), 512)]
        for i, chunk in enumerate(chunks):
            input_ids = chunk

            # Search the input_ids for the first instance of the `[SEP]` token.
            sep_index = input_ids.index(tokenizer.sep_token_id)

            # The number of segment A tokens includes the [SEP] token istelf.
            num_seg_a = sep_index + 1

            # The remainder are segment B.
            num_seg_b = len(input_ids) - num_seg_a

            # Construct the list of 0s and 1s.
            segment_ids = [0]*num_seg_a + [1]*num_seg_b

            assert len(segment_ids) == len(input_ids)

            output = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                                    return_dict=True)
            
            start_scores = output.start_logits
            end_scores = output.end_logits

            answer_start = torch.argmax(start_scores) + i * 512
            answer_end = torch.argmax(end_scores) + i * 512

            answer_dict[(answer_start, answer_end)] = answer_start + answer_end

        answer_start, answer_end = max(answer_dict, key=answer_dict.get)
        print(answer_start, answer_end)

    else:

        # Search the input_ids for the first instance of the `[SEP]` token.
        sep_index = input_ids.index(tokenizer.sep_token_id)

        # The number of segment A tokens includes the [SEP] token istelf.
        num_seg_a = sep_index + 1

        # The remainder are segment B.
        num_seg_b = len(input_ids) - num_seg_a

        # Construct the list of 0s and 1s.
        segment_ids = [0]*num_seg_a + [1]*num_seg_b

        assert len(segment_ids) == len(input_ids)

        output = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                                return_dict=True)
        
        start_scores = output.start_logits
        end_scores = output.end_logits

        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)

    # answer = ' '.join(tokens[answer_start:answer_end+1])
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    outputs[id] = answer

    # print(outputs)
    # print(f"Question: {question}")

# save predictions to a file
output_file = 'predict.json'
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(outputs, file, ensure_ascii=False, indent=4)

print(f"Predictions saved to {output_file}")

######################## Task 4.1 ########################

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

######################## Task 4.2 ########################

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

######################## Task 4.3 ########################

from langchain.chains import LLMChain, SequentialChain 
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
# from langchain_openai import OpenAI
from pprint import pprint
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model_name="claude-3-opus-20240229", anthropic_api_key="sk-")

# Language detection
prompt_language = PromptTemplate.from_template(
    'Please detect the language of the following text and reply with only the language name: {text}',
)
language_chain = LLMChain(llm=llm, prompt=prompt_language, output_key='language')

# Translation 
prompt_translate = ChatPromptTemplate.from_template(
    'Please translate the following {language} text to English: {text}',
)
translate_chain = LLMChain(llm=llm, prompt=prompt_translate, output_key='english_text')

# Sentiment analysis
prompt_sentiment = PromptTemplate.from_template(
    'What is the sentiment of the following text? Respond with Positive, Negative or Neutral: {english_text}',  
)
sentiment_chain = LLMChain(llm=llm, prompt=prompt_sentiment, output_key='sentiment')

# Generate reply based on sentiment
prompt_reply = PromptTemplate.from_template(
    """Write a reply to the following text based on its sentiment: 
    Sentiment: {sentiment}
    Text: {english_text}
    
    If the sentiment is Positive, give a short praise.  
    If Negative, give some comforting words.
    If Neutral, give a neutral acknowledgement.""",
)
reply_chain = LLMChain(llm=llm, prompt=prompt_reply, output_key='english_reply')

# Translate reply back to original language
prompt_backtranslate = PromptTemplate.from_template(
    'Please translate the following English reply to {language}: {english_reply}',
)
backtranslate_chain = LLMChain(llm=llm, prompt=prompt_backtranslate, output_key='origin_reply')

# overall_chain = SequentialChain(
#     chains=[
#         language_chain,
#         translate_chain,
#         sentiment_chain,
#         reply_chain, 
#         backtranslate_chain
#     ],
#     input_variables=['text'],
#     output_variables=['language','english_text','sentiment','english_reply','origin_reply'],
# )

def process_text(text):
    language = language_chain.run(text)
    print(f"Detected language: {language}")

    if language.lower() == 'english':
        english_text = text
    else:
        english_text = translate_chain.run(text=text, language=language)

    sentiment = sentiment_chain.run(english_text)

    english_reply = reply_chain.run(sentiment=sentiment, english_text=english_text) 

    if language.lower() == 'english':
        reply = english_reply
    else:  
        reply = backtranslate_chain.run(language=language, english_reply=english_reply)

    return {
        'original_text': text,
        'language': language, 
        'english_text': english_text,
        'sentiment': sentiment,
        'english_reply': english_reply,
        'final_reply': reply
    }

# # Example usage
sample_text1 = "This movie was amazing! The acting was superb and the plot kept me hooked throughout."
sample_text2 = "这部电影拍得真不错,演员演技在线,故事情节也很吸引人,推荐大家去看。"
# result = overall_chain(sample_text2)
# pprint(result)
result1 = process_text(sample_text1)
pprint(result1)

result2 = process_text(sample_text2) 
pprint(result2)