from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    number: float

class OutputData(BaseModel):
    result: float

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/process")
def process_data(data: InputData) -> OutputData:
    result = data.number ** 2
    return OutputData(result=result)
