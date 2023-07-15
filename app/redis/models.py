from pydantic import BaseModel
from datetime import datetime


class Message(BaseModel):
    msg: str
    timestamp: str = str(datetime.now())
