from transformers import pipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

def extract_ner(input_dict):
    result = []
    temp_ner_text = ""
    temp_ner_label = None  # Thêm biến để lưu label hiện tại

    for key, value in input_dict:

        # Nếu có @@ trong token thì text sẽ nối trực tiếp với nhau mà không có dấu cách
        space = " "
        if '@@' in key: space = "" 

        # Xóa '@@' khỏi key
        clean_key = key.replace('@@', '')

        # Nhãn 3 và 4 sẽ không có space bởi vì các token này sẽ kết hợp với nhau thành một chuỗi
        if value == 'LABEL_3':
            if temp_ner_text != "":
                temp_ner_text = temp_ner_text.strip(' ')
                result.append({'text': temp_ner_text, 'label': temp_ner_label})  # Lưu cả text và label
            temp_ner_text = ""
            temp_ner_text = temp_ner_text +  clean_key + space
            temp_ner_label = value  # Cập nhật label hiện tại
        if value == 'LABEL_4':
            temp_ner_text = temp_ner_text +  clean_key + space

        # Nhãn 1 và 2 có space bởi vì các token này sẽ kết hợp với nhau thành một cụm từ
        if value == 'LABEL_1':
            if temp_ner_text != "":
                temp_ner_text = temp_ner_text.strip(' ')
                result.append({'text': temp_ner_text, 'label': temp_ner_label})
            temp_ner_text = ""
            temp_ner_text = temp_ner_text +  clean_key + space
            temp_ner_label = value
        if value == 'LABEL_2':
            temp_ner_text = temp_ner_text +  clean_key + space

        # Nhãn 5 và 6 có space bởi vì các token này sẽ kết hợp với nhau thành một cụm từ
        if value == 'LABEL_5':
            if temp_ner_text != "":
                temp_ner_text = temp_ner_text.strip(' ')
                result.append({'text': temp_ner_text, 'label': temp_ner_label})
            temp_ner_text = ""
            temp_ner_text = temp_ner_text +  clean_key + space
            temp_ner_label = value
        if value == 'LABEL_6':
            temp_ner_text = temp_ner_text +  clean_key + space

        # Nhãn 7 và 8 có space bởi vì các token này sẽ kết hợp với nhau thành một cụm từ
        if value == 'LABEL_7':
            if temp_ner_text != "":
                temp_ner_text = temp_ner_text.strip(' ')
                result.append({'text': temp_ner_text, 'label': temp_ner_label})
            temp_ner_text = ""
            temp_ner_text = temp_ner_text +  clean_key + space
            temp_ner_label = value
        if value == 'LABEL_8':
            temp_ner_text = temp_ner_text +  clean_key + space

        # Nhãn 9 và 10 có space bởi vì các token này sẽ kết hợp với nhau thành một cụm từ
        if value == 'LABEL_9':
            if temp_ner_text != "":
                temp_ner_text = temp_ner_text.strip(' ')
                result.append({'text': temp_ner_text, 'label': temp_ner_label})
            temp_ner_text = ""
            temp_ner_text = temp_ner_text +  clean_key + space
            temp_ner_label = value
        if value == 'LABEL_10':
            temp_ner_text = temp_ner_text +  clean_key + space

        # Nhãn 11 và 12 có space bởi vì các token này sẽ kết hợp với nhau thành một cụm từ
        if value == 'LABEL_11':
            if temp_ner_text != "":
                temp_ner_text = temp_ner_text.strip(' ')
                result.append({'text': temp_ner_text, 'label': temp_ner_label})
            temp_ner_text = ""
            temp_ner_text = temp_ner_text +  clean_key + space
            temp_ner_label = value
        if value == 'LABEL_12':
            temp_ner_text = temp_ner_text +  clean_key + space

    if temp_ner_text != "" and temp_ner_label != None:
        temp_ner_text = temp_ner_text.strip(' ')
        result.append({'text': temp_ner_text, 'label': temp_ner_label})  # Lưu text cuối cùng cùng với label

    # Lọc result chỉ lấy những giá trị có label khác None
    result = [item for item in result if item['label'] is not None]

    # Trả về kết quả dưới dạng list chứa cả text và label
    return result

def truncate_sentence(sentence, max_words=100):
    words = sentence.split()  # Tách câu thành danh sách các từ
    if len(words) > max_words:  # Nếu số từ vượt quá max_words
        return " ".join(words[:max_words])  # Ghép lại chuỗi chỉ với max_words từ
    return sentence

def get_ner_from_sentence(sentence: str, ner_model, ner_tokenizer):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    sentence = truncate_sentence(sentence)
    nlp = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, device=device)
    ners = nlp(sentence)
    ner_extract = []
    for ner in ners:
        if ner['entity'] != 'LABEL_0':
            ner_tuple = (ner['word'], ner['entity'])
            #print(ner_tuple)
            ner_extract.append(ner_tuple)
    ner_extract = extract_ner(ner_extract)
    return ner_extract

#print(ner_extract)