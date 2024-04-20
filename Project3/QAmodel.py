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