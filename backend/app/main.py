import sys
import os


import pandas as pd
from fastapi import FastAPI
from app.feature_engineering.services.exp_eval import ExpressionEvaluator



app = FastAPI()


@app.get("/")
async def read_main():
    return {"msg": "Hello World"}


