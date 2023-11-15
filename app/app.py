from fastapi import FastAPI
from routers import root, predict, model

app = FastAPI()

app.include_router(root.router)
app.include_router(predict.router)
app.include_router(model.router)