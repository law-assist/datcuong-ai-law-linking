import torch
from Components.classification import PhoBERTForSTS, get_prediction
from transformers import AutoTokenizer, AutoModelForTokenClassification, RobertaForSequenceClassification
from Components.similarity import similarity_score
from utils.utils import is_similarity_available
from Components.ner import get_ner_from_sentence
from transformers import BertTokenizerFast, BertForSequenceClassification

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# similarity_tokenizer_V2 = AutoTokenizer.from_pretrained('vinai/phobert-base')
# similarity_model_V2 = PhoBERTForSTS().to(device)
# state_dict_v2 = torch.load("models/STS_model_V2/bert-sts-V2.pt")
# similarity_model_V2.load_state_dict(state_dict_v2)    


# sentence_1 = "Điều 11. Chế độ, chính sách đối với Trợ giúp viên pháp lý"
# sentence_2 = "a) Trợ giúp viên pháp lý;"

# sentence_tokenized_1 = similarity_tokenizer_V2(
#     sentence_1, padding='max_length', max_length=256, truncation=True, return_tensors="pt"
# ).to(device=device)

# sentence_tokenized_2 = similarity_tokenizer_V2(
#     sentence_2, padding='max_length', max_length=256, truncation=True, return_tensors="pt"
# ).to(device=device)

# score = similarity_score(sentence_tokenized_1, sentence_tokenized_2, similarity_model_V2)

# print("điểm tương đồng: ", score)


#-----------------------------------NER----------------------------------#

# ner_model = AutoModelForTokenClassification.from_pretrained("models/ner_model").to("cuda")
# ner_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# sentence = "QUYẾT ĐỊNH BAN HÀNH KẾ HOẠCH TRIỂN KHAI THI HÀNH LUẬT QUẢN LÝ, SỬ DỤNG VŨ KHÍ, VẬT LIỆU NỔ VÀ CÔNG CỤ HỖ TRỢ"

# ner = get_ner_from_sentence(sentence, ner_model, ner_tokenizer)

# print(ner)



# #-----------------------------------CLASSIFICATION----------------------------------#

# Set up classification model
model_classification = RobertaForSequenceClassification.from_pretrained("models/Classification_model", num_labels=6).to("cuda")
tokenizer_classification = AutoTokenizer.from_pretrained("models/Classification_model", do_lower_case=True)


description_refering_legislation_name = "Điều 1. Sửa đổi, bổ sung một số điều của Nghị định số 62/2015/NĐ-CP ngày 18 tháng 7 năm 2015 quy định chi tiết và hướng dẫn thi hành một số điều của Luật Thi hành án dân sự được sửa đổi, bổ sung một số điều theo Nghị định số 33/2020/NĐ-CP ngày 17 tháng 3 năm 2020 của Chính phủ như sau:"
flag_quote = False
#result = is_similarity_available(description_refering_legislation_name, tokenizer_classification, model_classification)
classification, flag_quote = get_prediction(text = description_refering_legislation_name,
                                            flag_quote = flag_quote, 
                                            tokenizer_classification = tokenizer_classification, 
                                            model_classification = model_classification)
print(classification)

# json_data_name = "NGHỊ ĐỊNH QUY ĐỊNH CHI TIẾT THI HÀNH MỘT SỐ ĐIỀU CỦA LUẬT BẢO VỆ TÀI NGUYÊN NƯỚC"

# is_similarity_flag, name_ners = is_similarity_available(json_data_name, 
#                                                 tokenizer_classification = tokenizer_classification, 
#                                                 model_classification = model_classification,
#                                                 tokenizer_ner = ner_tokenizer,
#                                                 model_ner = ner_model)

# print("result: ", is_similarity_flag)
# print("ner: ", name_ners[0]['text'])