import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Model artifacts and generated metrics load at startup; restart Flask after retraining.
model_dt = joblib.load('model_dt.pkl')
model_mlp = joblib.load('model_mlp.pkl')
scaler = joblib.load('scaler.pkl')

with open('medians.json', 'r') as f:
    train_medians = json.load(f)

with open('metrics.json', 'r', encoding='utf-8') as f:
    metrics_data = json.load(f)

FEATURE_NAMES = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

FEATURE_INFO = {
    'ph': {
        'name': 'Derajat Keasaman (pH)',
        'unit': 'pH (0-14)',
        'min': 0.0, 'max': 14.0, 'step': 0.1,
        'desc': 'Indikator tingkat keasaman atau kebasaan air.'
    },
    'Hardness': {
        'name': 'Kesadahan Air (Hardness)',
        'unit': 'mg/L',
        'min': 40.0, 'max': 350.0, 'step': 1.0,
        'desc': 'Kandungan garam kalsium dan magnesium terlarut dalam air.'
    },
    'Solids': {
        'name': 'Zat Padat Terlarut (TDS / Solids)',
        'unit': 'ppm (mg/L)',
        'min': 300.0, 'max': 65000.0, 'step': 100.0,
        'desc': 'Total mineral, garam, dan logam terlarut dalam air.'
    },
    'Chloramines': {
        'name': 'Kloramin (Chloramines)',
        'unit': 'ppm',
        'min': 0.0, 'max': 15.0, 'step': 0.1,
        'desc': 'Senyawa klorin dan amonia yang berfungsi sebagai disinfektan air.'
    },
    'Sulfate': {
        'name': 'Kadar Sulfat (Sulfate)',
        'unit': 'mg/L',
        'min': 100.0, 'max': 500.0, 'step': 1.0,
        'desc': 'Mineral alami dalam air; kadar tinggi menyebabkan rasa pahit & gangguan cerna.'
    },
    'Conductivity': {
        'name': 'Daya Hantar Listrik (Conductivity)',
        'unit': 'μS/cm',
        'min': 100.0, 'max': 800.0, 'step': 5.0,
        'desc': 'Kemampuan air menghantarkan arus listrik sejalan konsentrasi ion.'
    },
    'Organic_carbon': {
        'name': 'Karbon Organik Total (TOC)',
        'unit': 'ppm',
        'min': 2.0, 'max': 30.0, 'step': 0.1,
        'desc': 'Jumlah bahan organik terlarut yang berasal dari materi alami/limbah.'
    },
    'Trihalomethanes': {
        'name': 'Trihalometana (THM)',
        'unit': 'μg/L',
        'min': 0.0, 'max': 130.0, 'step': 1.0,
        'desc': 'Senyawa sampingan klorinasi air; berpotensi karsinogenik pada kadar tinggi.'
    },
    'Turbidity': {
        'name': 'Kekeruhan Air (Turbidity)',
        'unit': 'NTU',
        'min': 1.0, 'max': 7.0, 'step': 0.1,
        'desc': 'Tingkat kejernihan air akibat partikel tersuspensi.'
    }
}

PRESETS = {
    'potable': {
        'title': 'Sampel Air Layak Minum (Potable)',
        'desc': 'Data aktual dari dataset (index 1186) — kedua model memprediksi Potable dengan keyakinan tinggi.',
        'values': {
            'ph': 9.48,
            'Hardness': 122.91,
            'Solids': 56351.4,
            'Chloramines': 4.22,
            'Sulfate': 219.55,
            'Conductivity': 480.85,
            'Organic_carbon': 13.53,
            'Trihalomethanes': 41.73,
            'Turbidity': 4.13
        }
    },
    'non_potable': {
        'title': 'Sampel Air Tidak Layak (Non-Potable)',
        'desc': 'Data aktual dari dataset (index 76) — kedua model memprediksi Non-Potable dengan keyakinan tinggi.',
        'values': {
            'ph': 3.51,
            'Hardness': 158.73,
            'Solids': 23029.66,
            'Chloramines': 6.82,
            'Sulfate': 286.52,
            'Conductivity': 307.38,
            'Organic_carbon': 9.71,
            'Trihalomethanes': 79.85,
            'Turbidity': 3.76
        }
    },
    'standard_pam': {
        'title': 'Sampel Air Olahan Layak Minum',
        'desc': 'Data aktual dari dataset (index 1566) — pH netral, kedua model memprediksi Potable.',
        'values': {
            'ph': 7.7,
            'Hardness': 121.58,
            'Solids': 16978.93,
            'Chloramines': 6.81,
            'Sulfate': 354.25,
            'Conductivity': 368.22,
            'Organic_carbon': 13.0,
            'Trihalomethanes': 69.26,
            'Turbidity': 3.4
        }
    }
}

class PredictionInputError(ValueError):
    pass


def validate_prediction_input(features_dict, model_choice):
    if not isinstance(model_choice, str) or model_choice not in {'dt', 'mlp'}:
        raise PredictionInputError('Model harus dipilih sebagai Decision Tree atau MLP.')
    if not isinstance(features_dict, dict):
        raise PredictionInputError('Data fitur harus berupa objek dengan 9 parameter kualitas air.')

    missing_features = [feature for feature in FEATURE_NAMES if feature not in features_dict]
    if missing_features:
        raise PredictionInputError(
            'Parameter wajib belum diisi: ' + ', '.join(missing_features) + '.'
        )

    unexpected_features = sorted(set(features_dict) - set(FEATURE_NAMES))
    if unexpected_features:
        raise PredictionInputError(
            'Parameter tidak dikenal: ' + ', '.join(unexpected_features) + '.'
        )

    values = {}
    for feature in FEATURE_NAMES:
        raw_value = features_dict[feature]
        try:
            if isinstance(raw_value, bool):
                raise ValueError
            value = float(raw_value)
        except (TypeError, ValueError):
            raise PredictionInputError(f'Nilai {feature} harus berupa angka yang valid.') from None

        if not np.isfinite(value):
            raise PredictionInputError(f'Nilai {feature} harus berupa angka yang terbatas.')

        info = FEATURE_INFO[feature]
        if value < info['min'] or value > info['max']:
            raise PredictionInputError(
                f'{info["name"]} harus berada pada rentang {info["min"]:g} sampai {info["max"]:g} {info["unit"]}.'
            )
        values[feature] = value

    return values


def predict_sample(features_dict, model_choice='dt'):
    values = validate_prediction_input(features_dict, model_choice)
    row = [values[feature] for feature in FEATURE_NAMES]
    X_input = pd.DataFrame([row], columns=FEATURE_NAMES)
    
    if model_choice == 'mlp':
        X_scaled = scaler.transform(X_input)
        pred = model_mlp.predict(X_scaled)[0]
        prob = model_mlp.predict_proba(X_scaled)[0]
        model_name = "Multilayer Perceptron (MLP)"
    else:
        pred = model_dt.predict(X_input)[0]
        prob = model_dt.predict_proba(X_input)[0]
        model_name = "Decision Tree Classifier"
        
    prob_non_potable = round(float(prob[0]) * 100, 2)
    prob_potable = round(float(prob[1]) * 100, 2)
    
    vals = X_input.iloc[0].to_dict()
    return {
        'prediction': int(pred),
        'label': 'PREDIKSI MODEL: POTABLE (1)' if pred == 1 else 'PREDIKSI MODEL: NON-POTABLE (0)',
        'status_class': 'success' if pred == 1 else 'danger',
        'model_name': model_name,
        'prob_potable': prob_potable,
        'prob_non_potable': prob_non_potable,
        'predicted_probability': prob_potable if pred == 1 else prob_non_potable,
        'input_values': vals
    }

@app.route('/')
def index():
    return render_template(
        'index.html',
        metrics=metrics_data,
        feature_info=FEATURE_INFO,
        presets=PRESETS
    )

@app.route('/dataset')
def dataset():
    # Read first 15 rows of dataset for table preview
    df_preview = pd.read_csv('water_potability.csv').head(15).round(2).fillna('-')
    preview_data = df_preview.to_dict(orient='records')
    return render_template(
        'dataset.html',
        metrics=metrics_data,
        feature_info=FEATURE_INFO,
        preview_data=preview_data
    )

@app.route('/evaluasi')
def evaluasi():
    return render_template(
        'evaluasi.html',
        metrics=metrics_data
    )

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    model_choice = 'dt'
    error = None
    input_values = {f: train_medians[f] for f in FEATURE_NAMES}
    
    if request.method == 'POST':
        posted_values = {f: request.form.get(f, '') for f in FEATURE_NAMES}
        input_values.update(posted_values)
        model_choice = request.form.get('model_choice', 'dt')
        try:
            input_values = validate_prediction_input(posted_values, model_choice)
            result = predict_sample(input_values, model_choice=model_choice)
        except PredictionInputError as exc:
            error = str(exc)
            if model_choice not in {'dt', 'mlp'}:
                model_choice = 'dt'
        
    return render_template(
        'predict.html',
        feature_info=FEATURE_INFO,
        presets=PRESETS,
        input_values=input_values,
        model_choice=model_choice,
        result=result,
        error=error
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not request.is_json:
        return jsonify({'error': 'Kirim request dengan Content-Type application/json.'}), 415

    try:
        data = request.get_json()
    except BadRequest:
        return jsonify({'error': 'Body request bukan JSON yang valid.'}), 400
    if not isinstance(data, dict):
        return jsonify({'error': 'Body request harus berupa objek JSON.'}), 400

    model_choice = data.get('model_choice', 'dt')
    features = data.get('features')
    try:
        result = predict_sample(features, model_choice=model_choice)
    except PredictionInputError as exc:
        return jsonify({'error': str(exc)}), 400
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
