from typing import List
from tree import Tree
from Components.ner import get_ner_from_sentence
from Components.classification import is_similarity_available_classification
import gc, torch

def build_tree_from_json(data, ner_model, ner_tokenizer, model_classification, tokenizer_classification, similarity_tokenizer_V2, similarity_model_V2):

    torch.cuda.empty_cache()
    gc.collect()

    # Extract mainContent
    main_content_data = data['content']['mainContent']

    # Create a Tree and build it from mainContent
    legislation_id = data['numberDoc'].lower()
    legislation_name = data['name'].lower()
    db_id = data['_id'].lower()
    tree = Tree(legislation_id = legislation_id, 
                legislation_name = legislation_name, 
                db_id = db_id,
                ner_model = ner_model,
                ner_tokenizer = ner_tokenizer,
                model_classification = model_classification,
                tokenizer_classification = tokenizer_classification,
                similarity_tokenizer_V2 = similarity_tokenizer_V2,
                similarity_model_V2 = similarity_model_V2)
    
    tree.build_tree(main_content_data)
    print("Done building tree")
    return tree


def match_ner(refering_legislation: Tree, refered_legislations: List[Tree]):
    for refered_legislation in refered_legislations:
        refering_legislation.match_aggregation_ner(refered_legislation)
    return None

def is_similarity_available(text, tokenizer_classification, model_classification, tokenizer_ner, model_ner):
    return is_similarity_available_classification(text, tokenizer_classification, model_classification, tokenizer_ner, model_ner)

def flatten_and_unique(nested_list):
    unique_elements = list({item for sublist in nested_list for item in sublist})
    return unique_elements

def get_refered_legislation(description_data, mainContent_tree):
    ner_list = []
    i = 1
    while i < len(description_data):
        node = description_data[i]
        sentence = node["value"]
        if i == 1:
            sentence = description_data[0]["value"] + " " + description_data[1]["value"]

        internal_ner_extract_list = get_ner_from_sentence(sentence, 
                                            ner_model = mainContent_tree.ner_model, 
                                            ner_tokenizer = mainContent_tree.ner_tokenizer)
        node["internal_ner"] = [internal_ner_extract["text"].lower() for internal_ner_extract in internal_ner_extract_list]

        if node["internal_ner"] != []:
            ner_list.append(node["internal_ner"])

        i = i + 1
    return flatten_and_unique(ner_list)

def match_similarity(refering_legislation: Tree, refered_legislations: List[Tree]):
    for refered_legislation in refered_legislations:
        refering_legislation.match_similarity(refered_legislation)
    return None

def enhance_relation_laws_information_for_referring(refering_tree, relationLaws, json_data_referred_legislations):
    enhance_laws = []

    for referred_law in relationLaws:
        can_be_enhanced = False
        for json_data_referred_legislation in json_data_referred_legislations:
            document_name = json_data_referred_legislation['name'].lower()
            document_id = json_data_referred_legislation['numberDoc'].lower()
            referred_law = referred_law.lower()
            if referred_law in document_name or referred_law in document_id:
                classification_type_list = []
                classification_type_list = refering_tree.get_type_reference(document_name)
                classification_type_list = refering_tree.get_type_reference(document_id)

                if len(classification_type_list) == 0: classification_type_list.append(2)

                classification_type_list = list(set(classification_type_list))

                enhance_content = {"name": json_data_referred_legislation['name'], 
                                    "id": json_data_referred_legislation['_id'], 
                                    "numberDoc": json_data_referred_legislation['numberDoc'],
                                    "type": "referring",
                                    "classification": classification_type_list,
                                    "original_name": referred_law}
                if enhance_content not in enhance_laws:
                    enhance_laws.append(enhance_content)
                can_be_enhanced = True
                break

        if can_be_enhanced == False: 
            enhance_content = {"name": "", 
                                "id": "", 
                                "numberDoc": "",
                                "type": "",
                                "classification": [],
                                "original_name": referred_law}
            enhance_laws.append(enhance_content)        

    return enhance_laws

def enhance_referring_information_for_relation_laws(relationLaws_enhanced, json_data, json_data_referred_legislations):
    enhance_content = {"name": json_data['name'],
                        "id": json_data['_id'],
                        "numberDoc": json_data["numberDoc"],
                        "type": "referred",
                        "classification": []}

    for json_data_referred_legislation in json_data_referred_legislations:
        for refering_data in relationLaws_enhanced:
            if isinstance(refering_data, dict) and refering_data["id"].lower() == json_data_referred_legislation["_id"].lower():
                enhance_content["classification"] = refering_data["classification"]
                json_data_referred_legislation["relationLaws"].append(enhance_content)

    return enhance_content

def get_similarity_doc(infering_document_name_ner, json_data_referred_legislations):
    #infering_document_name_ner: luật đất đai
    result = []

    for json_data_referred_legislation in json_data_referred_legislations:
        document_name = json_data_referred_legislation['name'].lower()
        infering_document_name_ner_text = infering_document_name_ner.lower()
        if infering_document_name_ner_text == document_name:
               result.append(json_data_referred_legislation)

    return result
