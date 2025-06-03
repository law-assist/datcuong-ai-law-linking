from pymongo import MongoClient
from Components.ner import get_ner_from_sentence
from datetime import datetime
from bson.objectid import ObjectId
import re, os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
DATABASE_URI = os.getenv('DATABASE_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME')

def convert_document_from_db_to_available_json(document):
    document["_id"] = str(document["_id"])
    document["dateApproved"] = convert_datetime_to_string(document["dateApproved"])
    document["createdAt"] = convert_datetime_to_string(document["createdAt"])
    document["updatedAt"] = convert_datetime_to_string(document["updatedAt"])

    return document

def convert_document_from_json_to_db_instance(document):
    document["dateApproved"] = convert_string_to_datetime(document["dateApproved"])
    document["createdAt"] = convert_string_to_datetime(document["createdAt"])
    document["updatedAt"] = convert_string_to_datetime(document["updatedAt"])
    
    return document

def get_db_from_mongo(mongo_url: str):
    # Kết nối đến MongoDB
    client = MongoClient(mongo_url)

    # Truy cập cơ sở dữ liệu
    collection = client[DATABASE_NAME if DATABASE_NAME!= None else "law_linking"]["laws"]

    # Trả về danh sách các văn bản json
    return collection, client

def get_legislation_by_query(query): 
    # Lấy collection từ hàm get_db_from_mongo
    collection, client = get_db_from_mongo(DATABASE_URI)
    message = {"message": "", "data": ""}
    try:
        # Tìm tất cả tài liệu theo query
        legislations = collection.find(query)
        
        # Chuyển đổi kết quả thành danh sách
        legislations_list = [
            convert_document_from_db_to_available_json(doc) for doc in legislations
        ]
        
        if legislations_list:
            message = {"message": "success", "data": legislations_list}
        else:
            message = {"message": f"No legislations found with query {query}.", "data": []}
    except Exception as e:
        # Xử lý nếu có lỗi xảy ra
        message = {"message": f"An error occurred: {str(e)}"}

    client.close()
    
    return message


def update_legistration_after_linking(json_data, json_data_referred_legislations):
    # Lấy collection từ hàm get_db_from_mongo
    collection, client = get_db_from_mongo(DATABASE_URI)
    message = {"message": ""}
    
    try:
        lesgitration_id = ObjectId(json_data["_id"])
        query = {"_id": lesgitration_id}
        # json_data["_id"] = input_lesgitration_id
        updated_legistration = {i: json_data[i] for i in json_data if i!= "_id"}
        updated_legistration = convert_document_from_json_to_db_instance(updated_legistration)
        updated_legistration = {"$set": updated_legistration}
        collection.update_one(query, updated_legistration)
        
        for legistration in json_data_referred_legislations:
            referred_id = ObjectId(legistration["_id"])
            referred_query = {"_id": referred_id}
            # legistration["_id"] = referred_id
            updated_referred_legistration = {i: legistration[i] for i in legistration if i!="_id"}
            updated_referred_legistration = convert_document_from_json_to_db_instance(updated_referred_legistration)
            updated_referred_legistration = {"$set": updated_referred_legistration}
            collection.update_one(referred_query, updated_referred_legistration)
            
        message = {"message": "Update success"}
    except Exception as e:
        # Xử lý nếu có lỗi xảy ra
        message = {"message": f"An error occurred during DB update: {str(e)}"}
        
    client.close()
    
    return message

# Hàm chuyển đổi string ISO 8601 thành datetime
def ensure_datetime(date_value):
    if isinstance(date_value, datetime):
        return date_value
    elif isinstance(date_value, str):
        try:
            # Try format with fractional seconds
            return datetime.strptime(date_value, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            try:
                # Try format without fractional seconds
                return datetime.strptime(date_value, '%Y-%m-%dT%H:%M:%SZ')
            except ValueError:
                raise ValueError(f"Invalid date format for value: {date_value}")
    else:
        raise TypeError(f"Unsupported type for date: {type(date_value)}")
    
def convert_datetime_to_string(date_value):
    if isinstance(date_value, datetime):
        try:
            # Try format with fractional seconds
            return datetime.strftime(date_value, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            try:
                # Try format without fractional seconds
                return datetime.strftime(date_value, '%Y-%m-%dT%H:%M:%SZ')
            except ValueError:
                raise ValueError(f"Invalid value format for datetime: {date_value}")        
            
def convert_string_to_datetime(string_value):
    if isinstance(string_value, str):
        try:
            return datetime.strptime(string_value, '%Y-%m-%dT%H:%M:%S.%fZ')   
        except ValueError:
            try:
                return datetime.strptime(string_value, '%Y-%m-%dT%H:%M:%SZ')
            except ValueError:
                raise ValueError(f"Invalid value format for datetime: {string_value}")  
            

def filter_relative_law(relative_laws, refering_date, filter_attribute):
    # Lọc ra những văn bản có cùng tên nhưng khác date approve            
    # Tạo dictionary để nhóm các luật theo tên
    laws_grouped_by_name = {}
    for relative_law in relative_laws:
        rl_law_name = relative_law[filter_attribute]
        if rl_law_name not in laws_grouped_by_name:
            laws_grouped_by_name[rl_law_name] = []
        laws_grouped_by_name[rl_law_name].append(relative_law)

    # Chuyển đổi refering_date
    refering_date = ensure_datetime(refering_date)

    # Lọc ra luật có rl_dateApproved lớn nhất nhưng nhỏ hơn refering_date
    filtered_relative_laws = []
    for rl_law_name, laws in laws_grouped_by_name.items():
        
        valid_laws = [law for law in laws if ensure_datetime(law['dateApproved']) < refering_date]
        if valid_laws:
            # Tìm luật có rl_dateApproved lớn nhất trong các luật hợp lệ
            latest_law = max(valid_laws, key=lambda law: ensure_datetime(law['dateApproved']))
            filtered_relative_laws.append(latest_law)
    
    return filtered_relative_laws

# input ["Luật Tổ chức Chính phủ", "Luật Trợ giúp pháp lý"]
def matching_name_or_id_in_db(referred_laws_list, refering_date):
    relative_laws = []
    collection, client = get_db_from_mongo(DATABASE_URI)
    documents = collection.find()

    for referred_law in referred_laws_list:
        # Check có trùng tên hiện tại của luật đang xét không
        # Query 1: search by name
        query_name = {"name": re.compile(f"^{referred_law}$", re.IGNORECASE)}
        find_name_result_documents = get_legislation_by_query(query=query_name)
        
        # Query 2: search by numberDoc
        query_numberDoc = {"numberDoc": re.compile(f"^{referred_law}$", re.IGNORECASE)}
        find_numberDoc_result_documents = get_legislation_by_query(query=query_numberDoc)

        # # Merge results into relative_laws
        # relative_laws += find_name_result_documents.get("data", [])
        # relative_laws += find_numberDoc_result_documents.get("data", [])
        # Merge results into relative_laws
        for doc in find_name_result_documents.get("data", []):
            if doc not in relative_laws:
                relative_laws.append(doc)

        for doc in find_numberDoc_result_documents.get("data", []):
            if doc not in relative_laws:
                relative_laws.append(doc)


    print("----------------Legislation name found in database----------------")
    print("Relative laws before filtered: ", len(relative_laws))
    for relative_law in relative_laws:
        print(f"name: {relative_law['name']}, numberDoc: {relative_law['numberDoc']}")


    print("----------------Filter the relative laws----------------")

    relative_laws = filter_relative_law(relative_laws=relative_laws, refering_date=refering_date, filter_attribute="name")
    relative_laws = filter_relative_law(relative_laws=relative_laws, refering_date=refering_date, filter_attribute="numberDoc")

    print("Relative laws after filtered: ", len(relative_laws))

    for relative_law in relative_laws:
        print(f"name: {relative_law['name']}, numberDoc: {relative_law['numberDoc']}")

    # # --------------------  Run Debug  --------------------

    # for relative_law in relative_laws:
    #     if relative_law['name'] == "LUẬT KINH DOANH BẢO HIỂM":
    #         relative_laws = [relative_law]
    #         print("Đang debug chỉ lấy luật: ", relative_laws[0]['name'])
    #         break


    return relative_laws