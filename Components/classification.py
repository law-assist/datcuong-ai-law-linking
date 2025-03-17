import torch
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel
from Components.ner import get_ner_from_sentence
from Components.similarity import similarity_score

THRES_HOLD = 0.6

# class BertForSTS(torch.nn.Module):

#     def __init__(self):
#         super(BertForSTS, self).__init__()
#         self.bert = models.Transformer('bert-base-multilingual-cased', max_seq_length=512)
#         self.pooling_layer = models.Pooling(self.bert.get_word_embedding_dimension())
#         self.sts_bert = SentenceTransformer(modules=[self.bert, self.pooling_layer])

#     def forward(self, input_data):
#         output = self.sts_bert(input_data)['sentence_embedding']
#         return output
    
class PhoBERTForSTS(torch.nn.Module):
    def __init__(self):
        super(PhoBERTForSTS, self).__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")

    def forward(self, inputs_1, inputs_2):
        # Get embeddings
        encoded1 = self.phobert(**inputs_1).last_hidden_state
        encoded2 = self.phobert(**inputs_2).last_hidden_state

        # Calculate mean of token embeddings for each sentence
        mean_vector1 = encoded1.mean(dim=1)
        mean_vector2 = encoded2.mean(dim=1)

        # Calculate cosine similarity
        cosine_sim = torch.nn.CosineSimilarity(dim=1)
        similarity_score = cosine_sim(mean_vector1, mean_vector2)

        return similarity_score

def predict_similarity(sentence_pair, similarity_tokenizer, similarity_model):
  
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_input = similarity_tokenizer(sentence_pair, padding='max_length', max_length = 512, truncation=True, return_tensors="pt").to(device)
    test_input['input_ids'] = test_input['input_ids']
    test_input['attention_mask'] = test_input['attention_mask']
    del test_input['token_type_ids']
    output = similarity_model(test_input)
    sim = torch.nn.functional.cosine_similarity(output[0], output[1], dim=0).item()
    return sim

def get_prediction(text, flag_quote, tokenizer_classification, model_classification):
    # Trong trường hợp text bắt đầu bằng dấu ", thì ta có thể chắc chắn được kết quả trả về sẽ là 2, vì đây là nội dung của điều luật chỉnh sửa. Nội dung điều luật chỉnh sửa sẽ kết thúc khi gặp dấu ". Do đó ta sẽ có một lá cờ để kiểm soát điều này
    if not (text.startswith('“') and (text.endswith('”') or text.endswith('”.') or text.endswith('.”'))):
        if text.endswith('”') or text.endswith('”.') or text.endswith('.”'):
            flag_quote = False
            print("flag đang là false")
            print("-----------------")
            return 2, flag_quote
        if flag_quote:
            return 2, flag_quote
        if text.startswith('“'):
            flag_quote = True
            print("flag đang là true")
            print("-----------------")
            return 2, flag_quote

    lower_text = text.lower()

    # prepare our text into tokenized sequence
    inputs = tokenizer_classification(lower_text, padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
    # perform inference to our model
    outputs = model_classification(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)

    # make sure the input text that too short will be labeled as 2
    #five_label_similarity = "QUY ĐỊNH CHI TIẾT VÀ HƯỚNG DẪN THI HÀNH MỘT SỐ ĐIỀU CỦA LUẬT"
    if probs.argmax().item() == 5: #and predict_similarity(sentence_pair = (five_label_similarity, text), similarity_tokenizer = similarity_tokenizer, similarity_model = similarity_model) < THRES_HOLD:
      return 2, flag_quote

    # executing argmax function to get the candidate label
    return probs.argmax().item(), flag_quote


def is_similarity_available_classification(text, 
                                           tokenizer_classification, 
                                           model_classification, 
                                           tokenizer_ner, model_ner):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lower_text = text.lower()
    inputs = tokenizer_classification(lower_text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    outputs = model_classification(**inputs)
    probs = outputs[0].softmax(1)
    text_ners = get_ner_from_sentence(lower_text, model_ner, tokenizer_ner)

    #print("-------------debug-------------")
    #print("-------------debug-------------")

    if probs.argmax().item() == 5 and "sửa đổi, bổ sung" not in lower_text \
                                  and "sửa đổi" not in lower_text \
                                  and "bổ sung" not in lower_text \
                                  and len(text_ners) == 1 \
                                  and text_ners[0]["label"] == "LABEL_1":
        return True, text_ners
    return False, text_ners