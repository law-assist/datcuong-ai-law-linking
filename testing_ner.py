import re
from unidecode import unidecode

def convert_aggregation_ner_to_json_meaning(aggregation_ner):
    result = []
    for item in aggregation_ner:
        if isinstance(item, list):
            transformed_list = []
            for element in item:
                text = element['text']
                # Sử dụng biểu thức chính quy để tìm từ khóa và số đi kèm
                lower_text = text.lower()
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

aggregation_ner = [[{ "text": "điểm b", "label": "LABEL_9"}, {"text": "khoản 3","label": "LABEL_7"},{"text": "Điều 4","label": "LABEL_5"}]]
print(convert_aggregation_ner_to_json_meaning(aggregation_ner))