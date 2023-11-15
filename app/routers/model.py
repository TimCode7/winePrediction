from fastapi import APIRouter, HTTPException, Response
from model.WineCsvInput import WineCsvInput
from model.WineQualityPredictorTree import WineQualityPredictorTree
import pickle
from joblib import load
import pandas as pd


router = APIRouter()

predictor = load("data/wine_quality_model.joblib")


@router.get("/api/model")
async def get_model():
    """
    ### renvoie une version serializé du model, le modèle est téléchargeable
    """
    return Response(
        content=predictor.serialize(), media_type="application/octet-stream"
    )


@router.get("/api/model/description")
async def get_model_description():
    """
    ### Renvoie des informations sur le modèle :\n
    - paramètre du modèle\n
    - métrique de performance du modèle sur le jeu de test\n
    """
    return {"message": "model_description"}


@router.put("/api/model")
async def put_new_data(new_data: WineCsvInput):
    """
    ### Ajoute une nouvelle entrée de vin au fichier CSV.

    ### Cette route prend en entrée les caractéristiques d'un vin et les ajoute au fichier CSV avec un nouvel ID.
    """
    csv_file = "data/Wines.csv"
    try:
        df = pd.read_csv(csv_file)

        if df.empty:
            new_id = 1
        else:
            new_id = df["Id"].max() + 1

        new_row = new_data.model_dump(by_alias=True)
        new_row["Id"] = new_id
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(csv_file, index=False)
        return {"message": "Nouveau vin ajouté", "Id": new_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/model/retrain")
async def retrain():
    """
    ### Entraîne le modèle avec toutes les valeurs du fichiers csv.
    ### Enregistre le modèle.
    """
    try:
        predictor.train_model()
        predictor.save_model()
        return {"message": "Le modèle a été réentrainé avec succès"}, 200
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
