# @app.post("/reference_matching/")
# async def matching_legacy(file: UploadFile = File(...)):
#     start_time = time.time()
#     if file.content_type != "application/json":
#         return {"error": "File phải là định dạng JSON"}

#     file_contents = await file.read()

#     try:
#         json_data = json.loads(file_contents)
#     except json.JSONDecodeError:
#         return {"error": "File không phải là JSON hợp lệ"}
    
#     tree = build_tree_from_json(json_data)

#     relationLaws = get_refered_legislation(json_data['content']['description'], tree)

#     print("relationLaws: ", relationLaws)

#     end_time = time.time()

#     print(f"Thời gian chạy của API matching_ner: {end_time - start_time} giây")

#     return {"message": "Successful", "data": json_data}

    # json_data_referred_legislations = matching_name_or_id_in_db(relationLaws, json_data['dateApproved'], tree.ner_model, tree.ner_tokenizer)

    # if len(json_data_referred_legislations) == 0:
    #     json_data['content']['mainContent'] = tree.convert_to_json_format()
    #     json_data['relationLaws'] = relationLaws
    #     return {"message": "No legislation avaible in the database", "data": json_data, "update_data_in_db": []}



    # relationLaws_enhanced = enhance_relation_laws_information(relationLaws, json_data_referred_legislations, tree.ner_model, tree.ner_tokenizer)

    # #build tree for json_data_referred_legislations
    # tree_referred_legislations = [build_tree_from_json(json_data_referred_legislation) for json_data_referred_legislation in json_data_referred_legislations]

    # #Run match ner function
    # match_ner(tree, tree_referred_legislations)

    # #Save json_data of both tree and tree_referred_legislations to db
    # #Separate 2 input
    # if len(tree_referred_legislations) != 0:
    #     json_data['content']['mainContent'] = tree.convert_to_json_format()
    #     json_data['relationLaws'] = relationLaws_enhanced
    #     #save_to_json_file(json_data, "json_matching/ner/output/refering_result.json")

    # i = 0
    # json_string_referred_legislations = []
    # while i < len(tree_referred_legislations):
    #     json_data_referred_legislations[i]['content']['mainContent'] = tree_referred_legislations[i].convert_to_json_format()
    #     #save_to_json_file(json_data_referred_legislations[i], f"json_matching/ner/output/refered_result_{i}.json")

    #     #json_string_referred_legislation = convert_json_to_string(json_data_referred_legislations[i])

    #     json_string_referred_legislations.append(json_data_referred_legislations[i])
    #     #mermaid output
    #     #tree_referred_legislations[i].print_tree_mermaid(saving_path=f"json_matching/ner/output/mermaid/result_inferred_{i}.md")
    #     i = i + 1

    # # tree.print_tree_mermaid(saving_path="json_matching/ner/output/mermaid/result_inferring.md")
    
    # description_refering_legislation = json_data['content']['description']
    # description_refering_legislation_name_list = [a_description['value'] for a_description in description_refering_legislation]

    # flag_similarity = False
    # for description_refering_legislation_name in description_refering_legislation_name_list:
    #     word_count = len(description_refering_legislation_name.split())
    #     if word_count > 5 and is_similarity_available(description_refering_legislation_name, tree.tokenizer_classification, tree.model_classification):
    #         flag_similarity = True
    #         break

    # if not flag_similarity: print("not similarity case")

    # infering_document_name = json_data['name']
    # if flag_similarity:
    #     #lite version: json_matching/similarity/input/lite
    #     print("Detect Similarity, do similarity")
    #     json_data_referred_legislation = get_similarity_doc(infering_document_name, json_data_referred_legislations, tree.ner_model, tree.ner_tokenizer)
    #     if len(json_data_referred_legislation) != 0:
    #         json_data_referred_legislation = json_data_referred_legislation[0]
    #         print("Xây dựng cây cho similarity document")
    #         tree_referred_legislation = build_tree_from_json(json_data_referred_legislation)
    #         # Khởi chạy hàm match_similarity
    #         print("Khởi chạy hàm match_similarity")
    #         match_similarity(tree, [tree_referred_legislation])

    #         json_data['content']['mainContent'] = tree.convert_to_json_format()
    #         json_data['relationLaws'] = relationLaws_enhanced
    #         #lite version: json_matching/similarity/output/07-2007-NĐ-CP_refering_result_lite.json
    #         #save_to_json_file(json_data, "json_matching/similarity/output/07-2007-NĐ-CP_refering_result.json")

    #         json_data_referred_legislation['content']['mainContent'] = tree_referred_legislation.convert_to_json_format()
    #         #lite version: f"json_matching/similarity/output/refered_result_{i}_lite.json"
    #         #save_to_json_file(json_data_referred_legislation, f"json_matching/similarity/output/refered_result_{i}.json")
    
    # end_time = time.time()

    # print(f"Thời gian chạy của API matching_ner: {end_time - start_time} giây")
    #Không chạy similarity: 120 giây với mỗi 2 văn bản liên kết

    #return {"message": "Successful", "data": json_data, "update_data_in_db": json_string_referred_legislations}

# json_data = get_json_file("testing.json")

# tree = build_tree_from_json(json_data)

# relationLaws = get_refered_legislation(json_data['data']['content']['description'], tree)

# # print(relationLaws)
# # print(json_data['data']['dateApproved'])
# # print(type(json_data['data']['dateApproved']))

# result = matching_name_or_id_in_db(relationLaws, json_data['data']['dateApproved'], tree.ner_model, tree.ner_tokenizer)

# print("Ket qua: ", len(result))
# print("date approved of the first law: ", result[0]['dateApproved'])