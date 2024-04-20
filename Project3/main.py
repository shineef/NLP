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