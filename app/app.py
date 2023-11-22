from fastapi import FastAPI
from app.routers import root_router, predict, model

app = FastAPI()

app.include_router(root_router)
app.include_router(predict.router)
app.include_router(model.router)
