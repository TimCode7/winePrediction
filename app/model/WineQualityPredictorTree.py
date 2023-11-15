from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from joblib import dump
import pickle

from model.WineQualityInput import WineQualityInput


class WineQualityPredictorTree:
    def __init__(self, csv_file):
        """
        Initialise le modèle
        """
        self.data = pd.read_csv(csv_file)
        self.X = self.data.drop(columns=["quality", "Id"])
        self.y = self.data["quality"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.scaler = StandardScaler()
        self.X_train_normalized = self.scaler.fit_transform(self.X_train)
        self.X_test_normalized = self.scaler.transform(self.X_test)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train_model(self):
        """
        Entraîne le modèle
        """

        self.model.fit(self.X_train_normalized, self.y_train)

    def predict_quality(self, new_data: WineQualityInput):
        """
        Fait la prédiciton de la qualité sur des valeurs d'entrée
        """
        new_df = pd.DataFrame([new_data])
        new_data_normalized = self.scaler.transform(new_df)
        predicted_quality = self.model.predict(new_data_normalized)
        return predicted_quality[0]

    def predict_test(self):
        """
        Fait la prédiction sur les valeurs de test
        """

        y_pred = self.model.predict(self.X_test_normalized)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"R2 Score: {r2}")

    def save_model(self):
        """
        Sauvegarde le modèle dans un fichier .joblib
        """
        try:
            dump(self, "data/wine_quality_model.joblib")
            print("Modèle sauvegardé avec succès.")
        except Exception as e:
            print(f"{e}")

    def serialize(self):
        """
        retourne le modèle sérializé
        """
        return pickle.dumps(self)


# new_data = {
#     "fixed acidity": 7.5,
#     "volatile acidity": 0.685,
#     "citric acid": 0.07,
#     "residual sugar": 2.5,
#     "chlorides": 0.057999999999999996,
#     "free sulfur dioxide": 5.0,
#     "total sulfur dioxide": 9.0,
#     "density": 0.9963200000000001,
#     "pH": 3.38,
#     "sulphates": 0.55,
#     "alcohol": 10.9,
# }
