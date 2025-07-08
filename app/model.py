from pydantic import BaseModel
from typing import List


class EmailRequest(BaseModel):
    message: str


class BatchEmailRequest(BaseModel):
    messages: List[str]
