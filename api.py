from fastapi import FastAPI, File, UploadFile
import json
from Components.dto import InputDataRefMatchingJson, InputDataReferenceMatching
from utils.utils import build_tree_from_json, get_refered_legislation, match_ner, enhance_relation_laws_information_for_referring, match_similarity, is_similarity_available, get_similarity_doc, enhance_referring_information_for_relation_laws
from Components.databaseHandler import matching_name_or_id_in_db, get_legislation_by_query, update_legistration_after_linking
from Components.classification import PhoBERTForSTS, get_prediction
from transformers import AutoModelForTokenClassification, AutoTokenizer, RobertaForSequenceClassification
import time
import torch
from bson.objectid import ObjectId
from contextlib import asynccontextmanager
import uvicorn



# def get_json_file(json_name):
#     with open(json_name, 'r', encoding='utf-8') as file:
#         content = json.load(file)
#     return content

def load_model():

    # Set up ner model
    ner_model = AutoModelForTokenClassification.from_pretrained("models/ner_model").to("cuda")
    ner_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    print("Set up ner model")

    # Set up classification model
    model_classification = RobertaForSequenceClassification.from_pretrained("models/Classification_model", num_labels=6).to("cuda")
    tokenizer_classification = AutoTokenizer.from_pretrained("models/Classification_model", do_lower_case=True)
    print("Set up classification model")

    # Set up STS model V2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    similarity_tokenizer_V2 = AutoTokenizer.from_pretrained('vinai/phobert-base')
    similarity_model_V2 = PhoBERTForSTS().to(device)
    state_dict_v2 = torch.load("models/STS_model_V2/bert-sts-V2.pt")
    similarity_model_V2.load_state_dict(state_dict_v2)
    print("Set up STS model V2")

    return ner_model, ner_tokenizer, model_classification, tokenizer_classification, similarity_tokenizer_V2, similarity_model_V2

def convert_json_to_string(json_data):

    # Serialize dữ liệu JSON
    json_string = json.dumps(json_data, indent=4, ensure_ascii=False)

    return json_string

def save_to_json_file(json_data, json_save_name):
    
    json_string = convert_json_to_string(json_data)
    
    # Ghi vào file JSON
    with open(json_save_name, 'w', encoding='utf-8') as file:
        file.write(json_string)


ml_models = {}

@asynccontextmanager
async def preload_model(app: FastAPI):
        # Load ML model
        #Load model
        ner_model, ner_tokenizer, model_classification, tokenizer_classification, similarity_tokenizer_V2, similarity_model_V2 = load_model()
        ml_models["ner_model"] = ner_model
        ml_models["ner_tokenizer"] = ner_tokenizer
        ml_models["model_classification"] = model_classification
        ml_models["tokenizer_classification"] = tokenizer_classification
        ml_models["similarity_tokenizer_V2"] = similarity_tokenizer_V2
        ml_models["similarity_model_V2"] = similarity_model_V2
        yield


app = FastAPI(lifespan=preload_model)
    

@app.post("/reference_matching/id_input")
async def matching(data: InputDataReferenceMatching):
    legislation_id = data.input_string_id
    object_id = ObjectId(legislation_id)
    query = {"_id": object_id}
    legislation = get_legislation_by_query(query)

    if legislation["message"] != "success": return legislation

    # Vì query theo id nên kết quả chắc chắn chỉ có một và vì success nên chắc chắn có data
    legislation = legislation['data'][0]

    matching_api_result = matching_function(legislation, ml_models["ner_model"], ml_models["ner_tokenizer"], ml_models["model_classification"], ml_models["tokenizer_classification"], ml_models["similarity_tokenizer_V2"], ml_models["similarity_model_V2"])
    return matching_api_result

@app.post("/reference_matching/json_data")
async def matching(data: InputDataRefMatchingJson):
    legislation_id = data._id
    
    print(legislation_id)

def matching_function(json_data, ner_model, ner_tokenizer, model_classification, tokenizer_classification, similarity_tokenizer_V2, similarity_model_V2):
    start_time = time.time()

    # Build Tree structure from legislation json data from DB
    tree = build_tree_from_json(json_data, 
                                ner_model, 
                                ner_tokenizer, 
                                model_classification, 
                                tokenizer_classification, 
                                similarity_tokenizer_V2, 
                                similarity_model_V2)

    # Get infomation from description part (via NER model) and use it to find refered legislations
    description_data = json_data['content']['description']
    relationLaws = get_refered_legislation(description_data, tree)

    # Print relationLaws field of the input legislation json data
    print("relationLaws: ", relationLaws)

    # Find the referred legislations in DB
    json_data_referred_legislations = matching_name_or_id_in_db(relationLaws, json_data['dateApproved'])

    # Update relationsLaws field in the json format of the input legislation
    relationLaws_enhanced = enhance_relation_laws_information_for_referring(tree, relationLaws, json_data_referred_legislations)
    enhance_referring_information_for_relation_laws(relationLaws_enhanced, json_data, json_data_referred_legislations)
    
    # Rebuild Tree structure from updated json data (with relationLaws) and run it
    # Run it through both the classification model and NER model
    print("Build trees for referred...")
    tree_referred_legislations = [build_tree_from_json(json_data_referred_legislation, 
                                                       ner_model, ner_tokenizer, 
                                                       model_classification, 
                                                       tokenizer_classification, 
                                                       similarity_tokenizer_V2, 
                                                       similarity_model_V2) 
                                                       for json_data_referred_legislation in json_data_referred_legislations]
    
    # Update reference field at each node according to aggr_ner
    print("Do ner matching...")
    match_ner(tree, tree_referred_legislations)

    # print("Update data...")

    #update json data
    if len(json_data_referred_legislations) == 0:
        json_data['relationLaws'] = relationLaws
    else:
        json_data['relationLaws'] = relationLaws_enhanced
    json_data['content']['mainContent'] = tree.convert_to_json_format()

    #update tree data
    i = 0
    json_string_referred_legislations = []
    tree_with_legislation_name = []
    while i < len(tree_referred_legislations):
        json_data_referred_legislations[i]['content']['mainContent'] = tree_referred_legislations[i].convert_to_json_format()
        json_string_referred_legislations.append(json_data_referred_legislations[i])

        content = {"name": json_data_referred_legislations[i]['name'], "tree_data": tree_referred_legislations[i]}
        tree_with_legislation_name.append(content)

        i = i + 1
    json_data_referred_legislations = json_string_referred_legislations


    #------------------------------------------------DO SIMILARITY------------------------------------------------#

    is_similarity_flag, name_ners = is_similarity_available(json_data['name'], 
                                                 tokenizer_classification = tokenizer_classification, 
                                                 model_classification = model_classification,
                                                 tokenizer_ner = ner_tokenizer,
                                                 model_ner = ner_model)

    if not is_similarity_flag:
        print("văn bản không có tính similarity")

    if is_similarity_flag:
        print("Detect Similarity, thực hiện similarity")
        name_ner = name_ners[0]
        json_data_referred_similarity_legislations = get_similarity_doc(name_ner["text"], tree_with_legislation_name)
        
        if len(json_data_referred_similarity_legislations) > 1:
            return {"message": "json_data_referred_legislation chứa quá nhiều văn bản pháp luật để similarity", "data": json_data, "update_data_in_db": json_data_referred_legislations}
        elif len(json_data_referred_similarity_legislations) == 0:
            return {"message": "Không có văn bản để tiến hành similarity", "data": json_data, "update_data_in_db": json_data_referred_legislations}
        
        #Chắc chắn chỉ có một giá trị nên lấy giá trị tại 0
        json_data_referred_legislation = json_data_referred_similarity_legislations[0]

        print("Lấy cây cho văn bản similarity")
        print(json_data_referred_legislation)
        tree_referred_legislation = json_data_referred_legislation['tree_data']

        print("Khởi chạy hàm match_similarity")
        match_similarity(tree, [tree_referred_legislation])

        print("Lưu kết quả similarity")
        json_data['content']['mainContent'] = tree.convert_to_json_format()
        json_data['relationLaws'] = relationLaws_enhanced

        i = 0
        while i < len(tree_referred_legislations):
            if json_data_referred_legislations[i]["name"] == json_data_referred_legislation["name"]:
                print("Lưu kết quả mới cho văn bản được hướng dẫn")
                json_data_referred_legislations[i]['content']['mainContent'] = tree_referred_legislation.convert_to_json_format()
            i = i + 1

    update_message = update_legistration_after_linking(json_data, json_data_referred_legislations)
    print(f"Update database status: ", update_message["message"])

    end_time = time.time()
    print(f"Thời gian chạy của API matching_ner: {end_time - start_time} giây")

    return {"message": "Successful", "data": json_data, "update_data_in_db": json_data_referred_legislations}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)