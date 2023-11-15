from fastapi import APIRouter

router = APIRouter()


@router.get("/api/model")
async def get_model():
    return {"message": "model"}


@router.get("/api/model/description")
async def get_model_description():
    return {"message": "model description"}


@router.put("/api/model/retrain")
async def put_retrain():
    return {"message": "retrain put, ajoute une donnée supplémentaire"}


@router.post("/api/model/retrain")
async def retrain():
    return {"message": "retrain post"}
