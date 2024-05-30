import pandas as pd
from flask import Flask, request, jsonify, render_template
import xgboost as xgb

model_xgboost = xgb.Booster()
model_xgboost.load_model("xgboost.ubj")

if model_xgboost:
    print("Modelul XGBoost a fost încărcat cu succes.")
else:
    print("Eroare la încărcarea modelului XGBoost.")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_rating', methods=['POST'])
def predict_rating():
    new_app_features = request.json

    new_app = {
        'Rating Count': int(new_app_features['Rating Count']),
        'Minimum Installs': int(new_app_features['Minimum Installs']),
        'Maximum Installs': int(new_app_features['Maximum Installs']),
        'Price': float(new_app_features['Price']),
        'Ad Supported_True': int(new_app_features['Ad Supported_True'])
    }

    new_app_df = pd.DataFrame([new_app])
    new_app_dmatrix = xgb.DMatrix(new_app_df)
    predicted_rating = model_xgboost.predict(new_app_dmatrix)

    return jsonify({'predicted_rating': str(predicted_rating[0])})


if __name__ == '__main__':
    app.run(debug=True, port=2323)
