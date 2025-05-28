from pydantic import BaseModel, conlist

class AnswerListSchema(BaseModel):
    answers: conlist(str, min_length=1)

class RetrievalSchema(BaseModel):
    generated_query: str
    thoughts: str

