from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Root"}


@app.get("/api/predict")
async def get_predict():
    return {"message": "Hello World!"}


@app.post("/api/predict")
async def post_predict():
    return {"message": "predict"}


@app.get("/api/model")
async def get_model():
    return {"message": "model"}


@app.get("/api/model/description")
async def get_model_description():
    return {"message": "model description"}
