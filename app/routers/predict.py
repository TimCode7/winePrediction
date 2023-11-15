from fastapi import APIRouter

router = APIRouter()


@router.get("/api/predict")
async def get_predict():
    return {"message": "predict"}


@router.post("/api/predict")
async def post_predict():
    return {"message": "predict"}
