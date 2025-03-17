import json, torch, re
from Components.ner import get_ner_from_sentence
from Components.classification import PhoBERTForSTS, get_prediction
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import BertTokenizer
from Components.similarity import convert_embedding_json_to_tensor, similarity_score, extract_legislation_name_from_name_attr, THRESHOLD
from unidecode import unidecode
from copy import copy

# Define the Node and Tree classes
class Node:
    def __init__(self, name, value, unique_id, classification="", internal_ner=[], parent_ner=[], aggregation_ner=[], embedding="", parent=None, node_id=None):
        self.name = name
        self.value = value
        self.classification = classification
        self.internal_ner = internal_ner
        self.parent_ner = parent_ner
        self.aggregation_ner = aggregation_ner
        self.embedding = embedding
        self.parent = parent
        self.id = node_id
        self.unique_id = unique_id

        #Attributes that have initial value
        self.children = []
        self.address = []
        self.reference = []

    def get_ancestors(self):
        ancestors = []
        current = self.parent
        while current:
            if current.internal_ner:
                ancestors.append(current.internal_ner)
            current = current.parent
        return ancestors

    def get_info_address(self):
        ancestors = []
        current = self
        if current.name:
            while current:
                if current.name:
                    ancestors.append(current.name)
                current = current.parent
        return ancestors

    def add_child(self, child_node):
        child_node.parent = self  # Set the parent of the child node
        self.children.append(child_node)

    def self_matching(self, ner_val):
        # List of meaningful sequences
        meaningful_sequences = [
            [9, 7, 5, 1], [9, 7, 5, 3], [9, 7, 5, 11],
            [7, 5, 1], [7, 5, 3], [7, 5, 11],
            [5, 1], [5, 3], [5, 11],
            [1], [3], [11]
        ]

        # Resulting list of grouped NER values
        matched_sequences = []
        # Convert label values in ner_val to numerical form for comparison

        labels = [int(item["label"].split("_")[-1]) for item in ner_val]

        # Index to track current position in ner_val
        index = 0
        while index < len(labels):
            # Flag to check if we found a matching sequence
            found = False

            # Check for each meaningful sequence
            for sequence in meaningful_sequences:
                # Check if current part of labels matches the sequence
                if labels[index:index + len(sequence)] == sequence:
                    # Append the corresponding items from ner_val to matched_sequences
                    matched_sequences.append(ner_val[index:index + len(sequence)])
                    # Move the index forward by the length of the matched sequence
                    index += len(sequence)
                    found = True
                    break
            
            # If no sequence matched, add the current item as a single element
            if not found:
                matched_sequences.append(ner_val[index])
                index += 1
        
        return matched_sequences

    def remove_duplicates(self, array):
        unique_set = set()
        unique_array = []

        for item in array:
            if isinstance(item, list):  # Kiểm tra nếu item là danh sách lồng
                # Xử lý từng phần tử trong danh sách lồng, rồi chuyển thành tập hợp không trùng lặp
                unique_sub_array = [dict(t) for t in {tuple(d.items()) for d in item}]
                # Chuyển sub-array thành tuple để có thể thêm vào set kiểm tra trùng lặp
                sub_tuple = tuple(frozenset(d.items()) for d in unique_sub_array)
                if sub_tuple not in unique_set:
                    unique_set.add(sub_tuple)
                    unique_array.append(unique_sub_array)
            elif isinstance(item, dict):  # Kiểm tra nếu item là dictionary
                item_tuple = frozenset(item.items())
                if item_tuple not in unique_set:
                    unique_set.add(item_tuple)
                    unique_array.append(item)
            else:
                # Nếu item không phải là danh sách hoặc dictionary, có thể thêm các kiểu khác tùy yêu cầu
                if item not in unique_set:
                    unique_set.add(item)
                    unique_array.append(item)

        return unique_array

    def rearrange(self, array):
        # Danh sách meaningful_sequences đã cho
        meaningful_sequences = [
            [9, 7, 5, 1], [9, 7, 5, 3], [9, 7, 5, 11],
            [7, 5, 1], [7, 5, 3], [7, 5, 11],
            [5, 1], [5, 3], [5, 11],
            [1], [3], [11]
        ]
        
        def get_label_priority(label):
            # Lấy số ưu tiên từ nhãn
            return int(label.split('_')[1])
        
        def sort_by_meaningful_sequence(sub_array):
            # Lấy danh sách số ưu tiên từ các nhãn trong sub_array
            labels = [get_label_priority(item['label']) for item in sub_array]
            
            # Tìm thứ tự phù hợp trong meaningful_sequences
            for sequence in meaningful_sequences:
                if all(label in labels for label in sequence):
                    # Sắp xếp theo thứ tự của sequence
                    return sorted(sub_array, key=lambda x: sequence.index(get_label_priority(x['label'])))
            
            # Nếu không có thứ tự nào phù hợp, giữ nguyên thứ tự
            return sub_array

        # Sắp xếp từng mảng con
        result = [sort_by_meaningful_sequence(sub_array) for sub_array in array]
        
        return result

    def convert_aggregation_ner_to_json_meaning(self, aggregation_ner):
        result = []
        for item in aggregation_ner:
            if isinstance(item, list):
                transformed_list = []
                for element in item:
                    text = element['text']
                    lower_text = text.lower()
                    
                    # Sử dụng biểu thức chính quy để tìm từ khóa và số đi kèm
                    match = re.match(r"(khoản|điều|điểm|mục|chương)\s*(\d+|[a-z])", lower_text, re.IGNORECASE)
                    
                    if match:
                        # Chuyển đổi thành dạng yêu cầu (viết thường, không dấu cách)
                        prefix = match.group(1).lower()  # chuyển đổi từ khóa về dạng viết thường
                        number = match.group(2)  # lấy số
                        
                        # Loại bỏ dấu tiếng Việt
                        prefix = unidecode(prefix)
                        
                        transformed_text = f"{prefix}{number}"  # ghép lại thành từ hoàn chỉnh
                        transformed_list.append(transformed_text)
                    else:
                        # Giữ nguyên các mục không cần chuyển đổi
                        transformed_list.append(text)
                result.append(transformed_list)
            else:
                # Nếu item không phải là list, thêm trực tiếp vào kết quả
                result.append(item['text'])
        
        return result

    def ancestor_matching(self):
        # Step 1: Apply self_matching to obtain matched sequences
        initial_matches = self.self_matching(self.internal_ner)

        # Step 2: Separate direct matches (lists) and remaining non-list items
        final_result = [seq for seq in initial_matches if isinstance(seq, list)]
        remaining_items = [item for item in initial_matches if not isinstance(item, list)]

        # Step 3: Process each remaining item by merging with ancestor NERs
        iterator_remaining_items = []
        for item in remaining_items:
            iterator_remaining_items.extend([item])
            merged = copy(iterator_remaining_items)
            found_match = False

            # Go through each ancestor's NER values
            for ancestor_ner in self.parent_ner:
                merged.extend(ancestor_ner)  # Sequentially merge with ancestor NER
                merged_matched = self.self_matching(merged)  # Check for meaningful sequences

                # If a meaningful sequence is found, add to the result and stop merging
                if any(isinstance(m, list) for m in merged_matched):
                    for m in merged_matched:
                        if isinstance(m, list) and item in m:
                            final_result.append(m)
                            #found_match = True
                            break
                if found_match:
                    break

        final_result = self.post_processing(final_result)
        return final_result

    def post_processing(self, aggregation_ner):
        #Remove duplicate of the aggregation_ner
        aggregation_ner = self.remove_duplicates(aggregation_ner)

        #Make sure aggregation_ner have right order
        aggregation_ner = self.rearrange(aggregation_ner)

        aggregation_ner = self.convert_aggregation_ner_to_json_meaning(aggregation_ner)

        aggregation_ner = [[ner.lower() for ner in ners] for ners in aggregation_ner]

        return aggregation_ner

    def get_embedding(self, sentence, tokenizer):
        sentence_tokenized = tokenizer(
            sentence, padding='max_length', max_length=256, truncation=True, return_tensors="pt"
        )
        inputs_1 = {
            'input_ids': str(sentence_tokenized['input_ids'].squeeze(1).tolist()),
            'attention_mask': str(sentence_tokenized['attention_mask'].squeeze(1).tolist())
        }

        return inputs_1

    def enhance_information(self, flag_quote, ner_model, ner_tokenizer, model_classification, tokenizer_classification, 
                            similarity_tokenizer_V2):
        self.internal_ner = get_ner_from_sentence(self.value, ner_model, ner_tokenizer)
        self.parent_ner = self.get_ancestors()
        self.aggregation_ner = self.ancestor_matching()
        self.classification, flag_quote = get_prediction(text = self.value,
                                                         flag_quote = flag_quote, 
                                                         tokenizer_classification = tokenizer_classification, 
                                                         model_classification = model_classification)

        self.address = self.get_info_address()

        self.embedding = self.get_embedding(sentence = self.value, 
                                            tokenizer = similarity_tokenizer_V2)

        return flag_quote


    def convert_to_json_format(self):
        """Recursively converts the node and its children to a dictionary format suitable for JSON."""
        # Base dictionary structure
        node_dict = {
            "name": self.name,
            "value": self.value,
            "classification": self.classification,
            "internal_ner": self.internal_ner,
            "parent_ner": self.parent_ner,
            "aggregation_ner": self.aggregation_ner,
            "embedding": self.embedding,
            "reference": self.reference
        }

        # Add "content" key only if there are children
        if self.children:
            node_dict["content"] = [child.convert_to_json_format() for child in self.children]

        return node_dict

class Tree:
    def __init__(self, legislation_id, legislation_name, db_id, ner_model, ner_tokenizer, model_classification, tokenizer_classification, similarity_tokenizer_V2, similarity_model_V2):
        self.root = None
        self.legislation_id = legislation_id
        self.legislation_name = legislation_name
        self.db_id = db_id
        self.classification_flag_quote = False
        self.node_id_counter = 0
        self.unique_id_node = 0

        # Set up ner model
        self.ner_model = ner_model
        self.ner_tokenizer = ner_tokenizer

        # Set up classification model
        self.model_classification = model_classification
        self.tokenizer_classification = tokenizer_classification
    
        # Set up STS model V2
        self.similarity_tokenizer_V2 = similarity_tokenizer_V2
        self.similarity_model_V2 = similarity_model_V2     


    def build_tree(self, main_content_data):
        # Iterate over each item in main_content_data to build the root and children
        if self.root is None:
            self.root = Node(
                            name=None,
                            value=None,
                            unique_id="mermaid" + str(self.unique_id_node)
                        )
            self.unique_id_node = self.unique_id_node + 1

        for item in main_content_data:
            node = self._build_node(item)
            self.root.add_child(node)

    def _build_node(self, data, parent=None):
        # Create a Node with name, value, and parent
        node = Node(
            name=data.get("name"),
            value=data.get("value"),
            classification=data.get("classification"),
            internal_ner=data.get("internal_ner"),
            parent_ner=data.get("parent_ner"),
            aggregation_ner=data.get("aggregation_ner"),
            embedding=data.get("embedding"),
            parent=parent,
            node_id=self.node_id_counter,
            unique_id="mermaid" + str(self.unique_id_node)
        )

        self.unique_id_node = self.unique_id_node + 1
        self.node_id_counter = self.node_id_counter + 1
        self.classification_flag_quote = node.enhance_information(ner_model = self.ner_model,
                                                                  ner_tokenizer = self.ner_tokenizer,
                                                                  flag_quote = self.classification_flag_quote,
                                                                  model_classification = self.model_classification,
                                                                  tokenizer_classification = self.tokenizer_classification,
                                                                  similarity_tokenizer_V2 = self.similarity_tokenizer_V2)

        # Recursively add children if present
        if "content" in data:
            self.node_id_counter = 0
            for child_data in data["content"]:
                child_node = self._build_node(child_data, node)  # Set node as parent
                node.add_child(child_node)

        #print("Add node: ", node.value)

        return node
    
    def add_reference(self, refering_node, refering_id):
        # Check if there is a root node to start from
        if self.root is None:
            print("Tree is empty.")
        else:
            self._add_reference(self.root, refering_node, refering_id)

    def _add_reference(self, node, refering_node, refering_id):
        #TO DO
        full_address_with_id = node.address + [self.legislation_id]
        full_address_with_name = node.address + [self.legislation_name]

        for an_referencing_ner in refering_node.aggregation_ner:
            if len(an_referencing_ner) == 1:
                referencing_address = an_referencing_ner[0]
                # print("debug 1: ", referencing_address)
                # print("debug 2: ", self.legislation_name)
                if referencing_address.lower() == self.legislation_name or referencing_address.lower() == self.legislation_id:
                    new_reference = {"lawId": self.db_id, "LawRef": [], "index": None, "classification": refering_node.classification}
                    if new_reference not in refering_node.reference:
                        refering_node.reference += [new_reference]
            elif an_referencing_ner == full_address_with_id or an_referencing_ner == full_address_with_name:
                if node.address != []:
                    new_reference = {"lawId": self.db_id, "LawRef": list(reversed(node.address + ["mainContent"])), "index": node.id, "type": "referring", "classification": refering_node.classification}
                    if new_reference not in refering_node.reference:
                        refering_node.reference += [new_reference]
                if refering_node.address != []:
                    new_reference = {"lawId": refering_id, "LawRef": list(reversed(refering_node.address + ["mainContent"])), "index": refering_node.id, "type": "referred", "classification": refering_node.classification}
                    if new_reference not in node.reference:
                        node.reference += [new_reference]        


        for child in node.children:
            self._add_reference(child, refering_node, refering_id)


    #CODE for referencing
    def match_aggregation_ner(self, refered_legislation):
        # Check if there is a root node to start from
        if self.root is None:
            print("Tree is empty.")
        else:
            self._match_aggregation_ner(self.root, refered_legislation)

    def _match_aggregation_ner(self, node, refered_legislation):
        #TO DO
        if node.aggregation_ner != []:
            refered_legislation.add_reference(node, self.db_id)

        for child in node.children:
            self._match_aggregation_ner(child, refered_legislation)


    #CODE for similarity add reference
    def add_reference_for_similarity(self, refering_node, refering_id):
        if refering_node.embedding != "":
            embedding_refering = convert_embedding_json_to_tensor(refering_node.embedding)

            # Check if there is a root node to start from
            if self.root is None:
                print("Tree is empty.")
            else:
                self.max_score = float('-inf')
                self.node_with_max_score = None
                self._add_reference_for_similarity(self.root, refering_node, embedding_refering)
                if self.node_with_max_score != None:
                    if self.node_with_max_score.address != []:
                        new_reference = {"lawId": self.db_id, "LawRef": list(reversed(self.node_with_max_score.address + ["mainContent"])), "index": self.node_with_max_score.id, "type": "referring", "classification": 6}
                        if new_reference not in refering_node.reference:
                            refering_node.reference += [new_reference]
                    if refering_node.address != []:
                        new_reference = {"lawId": refering_id, "LawRef": list(reversed(refering_node.address + ["mainContent"])), "index": refering_node.id, "type": "referred", "classification": 6}
                        if new_reference not in self.node_with_max_score.reference:
                            self.node_with_max_score.reference += [new_reference]

    def _add_reference_for_similarity(self, node, refering_node, embedding_refering):
        #TO DO
        #node: luat phap ly
        #refereing node: nghi dinh bo sung chi tiet luat phap ly
        if node.embedding != "" and not node.value.endswith(":"):
            embedding_refered = convert_embedding_json_to_tensor(node.embedding)
            score = similarity_score(embedding_refering, embedding_refered, self.similarity_model_V2)

            print("---------------------Debug Similarity----------------------")
            print("refering_node value: ", refering_node.value)
            print("node value: ", node.value)
            print("score: ", score)

            if score > THRESHOLD and score > self.max_score:
                self.max_score = score
                self.node_with_max_score = node

        for child in node.children:
            self._add_reference_for_similarity(child, refering_node, embedding_refering)

    #CODE for similarity matching
    def match_similarity(self, refered_legislation):
        # Check if there is a root node to start from
        if self.root is None:
            print("Tree is empty.")
        else:
            self._match_similarity(self.root, refered_legislation)

    def _match_similarity(self, node, refered_legislation):
        #TO DO
        Black_list = ['Phạm vi điều chỉnh', 
                      'Đối tượng áp dụng', 
                      'Giải thích từ ngữ', 
                      "Hiệu lực thi hành", 
                      "Trách nhiệm thi hành", 
                      "Điều khoản chuyển tiếp"]
        black_list_flag = False
        if node.name != None and "dieu" in node.name:
            for black_list_sentence in Black_list:
                if black_list_sentence.lower() in node.value.lower():
                    black_list_flag = True
            if not black_list_flag: refered_legislation.add_reference_for_similarity(node, self.db_id)

        for child in node.children:
            self._match_similarity(child, refered_legislation)    

    def get_type_reference(self, document_numberDoc_or_name):
        # Check if there is a root node to start from
        classification_type_list = []
        if self.root is None:
            return classification_type_list
        else:
            self._get_type_reference(self.root, classification_type_list, document_numberDoc_or_name)
            return classification_type_list

    def _get_type_reference(self, node, classification_type_list, document_numberDoc_or_name):

        #TO DO
        for an_aggregation_ner_group in node.aggregation_ner:
            for a_ner in an_aggregation_ner_group:
                if a_ner.lower() == document_numberDoc_or_name.lower():
                    if node.classification not in classification_type_list:
                        if node.classification == 0 or node.classification == 5:
                            classification_type_list.append(2)
                        else:
                            classification_type_list.append(node.classification)

        for child in node.children:
            self._get_type_reference(child, classification_type_list, document_numberDoc_or_name)

    def print(self):
        # Check if there is a root node to start from
        if self.root is None:
            print("Tree is empty.")
        else:
            self._print_recursive(self.root, 0)

    def _print_recursive(self, node, level):
        # Print the current node's name and value with indentation based on level
        indent = " " * (level * 4)
        #parent_value = node.parent.value if node.parent else None
        #print(f"{indent}Parent Node: {parent_value}")
        if node.name:
            print(f"{indent}{node.name}: {node.value}")
        else:
            print(f"{indent}{node.value}")
        print("---------------------------------------------------------------")
        # print(f"{indent}Classification: {node.classification}")
        # print(f"{indent}Internal Ner: {node.internal_ner}")
        # print(f"{indent}Parent Ner: {node.parent_ner}")
        #print(f"{indent}Reference: {node.reference}")
        #print(f"{indent}Id: {node.id}")
        print(f"{indent}Embedding: {node.embedding}")
        print("---------------------------------------------------------------")        
        #print(f"{indent}Parent Ner: {node.parent_ner}")

        
        # Recursively print each child node, increasing the level
        for child in node.children:
            self._print_recursive(child, level + 1)

    def convert_to_json_format(self):
        """Converts the entire tree structure to a JSON formatted string."""
        if self.root:
            tree_dict = [children.convert_to_json_format() for children in self.root.children]
        else:
            tree_dict = []

        return tree_dict
    
    def print_tree_mermaid_recursive(self, node, tree_content, counter):
        preprocessed_content = ""
        if node.value:
            preprocessed_content = re.sub(r'\d+\.', '', node.value)
        
        #name = ""
        #if node.name != None:
        #    name = node.name
        
        one_node_content_detail = (f"{node.unique_id}[\"`Content: {preprocessed_content}\n Classification: {node.classification}\n Aggregation ner: {node.aggregation_ner if node.aggregation_ner else None}\n Reference: {node.reference}`\"]\n")
        one_node_content_link = ""
        if node.parent:
            one_node_content_link = f"{node.parent.unique_id}-->{node.unique_id}\n"
        
        # Thêm nội dung vào list thay vì cộng chuỗi
        tree_content.append(one_node_content_detail)
        tree_content.append(one_node_content_link)
        
        for child in node.children:
            counter = counter + 1
            self.print_tree_mermaid_recursive(child, tree_content, counter)
        
        return tree_content

    def print_tree_mermaid(self, saving_path):
        mermaid_string = ("```mermaid\n"
                        "---\n"
                        "title: Legislation Chart Tree\n"
                        "---\n"
                        "flowchart TD\n")
        
        # Sử dụng list để lưu các phần nội dung của cây
        tree_content = []
        tree_content = self.print_tree_mermaid_recursive(self.root, tree_content, 0)

        # Nối các phần tử trong list thành chuỗi
        content = mermaid_string + ''.join(tree_content)

        # Ghi nội dung vào file markdown 
        with open(saving_path, "w", encoding="utf-8") as file:
            file.write(content)

        return None
    
