from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd

class WineQualityPredictor:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.drop(columns=["quality", "Id"])  # Features
        self.y = self.data["quality"]  # Cible
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.scaler = StandardScaler()
        self.X_train_normalized = self.scaler.fit_transform(self.X_train)
        self.X_test_normalized = self.scaler.transform(self.X_test)
        self.model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
    
    def train_model(self):
        self.model.fit(self.X_train_normalized, self.y_train)
    
    def predict_quality(self, new_data):
        new_df = pd.DataFrame([new_data])
        new_data_normalized = self.scaler.transform(new_df)
        predicted_quality = self.model.predict(new_data_normalized)
        return predicted_quality[0]
    
    def predict_test(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print(mse)

# Usage :
predictor = WineQualityPredictor("data/Wines.csv")
predictor.train_model()
predictor.predict_test()

# Pour prédire la qualité d'un nouveau vin :
new_data = {
    'fixed acidity': 7.9,
    'volatile acidity': 0.35,
    'citric acid': 0.46,
    'residual sugar': 3.6,
    'chlorides': 0.078,
    'free sulfur dioxide': 15.0,
    'total sulfur dioxide': 37.0,
    'density': 0.9973,
    'pH': 3.35,
    'sulphates': 0.86,
    'alcohol': 12.8
}

predicted_quality = predictor.predict_quality(new_data)
print("La qualité prédite du vin est :", predicted_quality)
