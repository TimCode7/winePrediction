from fastapi import APIRouter

from model.WineQualityInput import WineQualityInput
from model.WineQualityPredictorTree import WineQualityPredictorTree
from joblib import load
from fastapi.responses import JSONResponse

router = APIRouter()


predictor = load("data/wine_quality_model.joblib")


@router.get("/api/predict")
async def get_predict():
    """
    ### Renvoie le meilleur vin.
    """
    return {"message": "predict"}


@router.post("/api/predict")
async def post_predict(input_data: WineQualityInput):
    """
    ### Renvoie la qualité prédite par le modèle avec les valeurs entrées
    """
    new_data = input_data.model_dump(by_alias=True)
    predicted_quality = predictor.predict_quality(new_data)
    return JSONResponse(predicted_quality)
