import torch
import ast
from Components.ner import get_ner_from_sentence

THRESHOLD = 0.7

def extract_legislation_name_from_name_attr(legislation_name_attr, ner_model, ner_tokenzier):
    result = ""
    legislation_ner = get_ner_from_sentence(legislation_name_attr, ner_model, ner_tokenzier)
    if len(legislation_ner) != 0:
        result = legislation_ner[0]['text']
    return result
    

def check_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

def convert_embedding_json_to_tensor(embedding):
    device = check_cuda()
    input_ids_str = embedding['input_ids']
    attention_mask_str = embedding['attention_mask']
    
    # Convert the string to lists
    input_ids_list = ast.literal_eval(input_ids_str)
    attention_mask_list = ast.literal_eval(attention_mask_str)

    # Convert the lists back to tensors
    input_ids_tensor = torch.tensor(input_ids_list)
    attention_mask_tensor = torch.tensor(attention_mask_list)

    tensor_embedding = {
        'input_ids': input_ids_tensor.squeeze(1).to(device),
        'attention_mask': attention_mask_tensor.squeeze(1).to(device)
    }

    return tensor_embedding
    

def similarity_score(tokenized_sentence_1, tokenized_sentence_2, model):
    with torch.no_grad():  # Disable gradient tracking for inference
        similarity_score = model(tokenized_sentence_1, tokenized_sentence_2)
        return similarity_score.item()
    return -1