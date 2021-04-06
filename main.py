from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json

import torch
from transformers import BertForSequenceClassification, BertTokenizer


app = FastAPI()

@app.get('/')
def show():
    return {"text": "Hello World"}


@app.get('/generate/')
def show():
    return {"Page": "show page"}

@app.post('/generate/')
def generate(input):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cpu') 
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load('./models/bert_statedict', map_location=device))
    model.eval()
    test_encoding = tokenizer([json.loads(input)['passage']], return_tensors='pt', padding=True, truncation=True)
    test_input_ids = test_encoding['input_ids']
    test_attention_mask = test_encoding['attention_mask']
    prob = torch.nn.functional.softmax(model(test_input_ids, test_attention_mask).logits, dim=1)[0][1]
    return {"prob": float(prob)}

    
