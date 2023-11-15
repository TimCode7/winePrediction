from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import pickle


class WineQualityPredictor:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.drop(columns=["quality", "Id"])
        self.y = self.data["quality"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print(self.X_test.head())
        self.scaler = StandardScaler()
        self.X_train_normalized = self.scaler.fit_transform(self.X_train)
        self.X_test_normalized = self.scaler.transform(self.X_test)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train_model(self):
        self.model.fit(self.X_train_normalized, self.y_train)

    def predict_quality(self, new_data):
        new_df = pd.DataFrame([new_data])
        new_data_normalized = self.scaler.transform(new_df)
        predicted_quality = self.model.predict(new_data_normalized)
        return predicted_quality[0]

    def predict_test(self):
        y_pred = self.model.predict(self.X_test_normalized)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"R2 Score: {r2}")


# Usage :
predictor = WineQualityPredictor("data/Wines.csv")
predictor.train_model()
try:
    with open("data/wine_quality_model.pickle", "wb") as file:
        pickle.dump(predictor.model, file)
    print("Modèle sauvegardé avec succès.")
except Exception as e:
    print(f"{e}")
predictor.predict_test()

# Pour prédire la qualité d'un nouveau vin :
new_data = {
    "fixed acidity": 6.9,
    "volatile acidity": 0.84,
    "citric acid": 0.21,
    "residual sugar": 4.1,
    "chlorides": 0.074,
    "free sulfur dioxide": 16.0,
    "total sulfur dioxide": 65.0,
    "density": 0.99842,
    "pH": 3.53,
    "sulphates": 0.72,
    "alcohol": 9.233333,
}

predicted_quality = predictor.predict_quality(new_data)
print("La qualité prédite du vin est :", predicted_quality)
