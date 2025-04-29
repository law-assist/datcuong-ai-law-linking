from pydantic import BaseModel


class InputDataReferenceMatching(BaseModel):
    input_string_id: str
  
class InputDataRefMatchingJson(BaseModel):
    id: str
    isDeleted: bool
    name: str
    category: str
    department: str
    baseURL: str
    pdfURL: str
    numberDoc: str
    dateApproved: str
    fields: list | None = []
    content: dict
    relationLaws: list | None = []
    createdAt: str
    updatedAt: str
    