from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
import joblib
import numpy as np
import shap
import os

app = Flask(__name__)


base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'static', 'fetal_health_model.pkl')
scaler_path = os.path.join(base_dir, 'static', 'scaler.pkl')

model = joblib.load(open(model_path, 'rb'))
sc = joblib.load(open(scaler_path, 'rb'))
explainer = shap.Explainer(model)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:

            # Collect inputs
            inputs = [float(request.form[field]) for field in [
                            'baseline_value', 'accelerations', 'fetal_movement', 'uterine_contractions',
                            'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
                            'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
                            'percentage_of_time_with_abnormal_long_term_variability',
                            'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min',
                            'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes',
                            'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance',
                            'histogram_tendency'
                        ]]

            feature_names = [
                'baseline_value', 'accelerations', 'fetal_movement', 'uterine_contractions',
                'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
                'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
                'percentage_of_time_with_abnormal_long_term_variability',
                'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min',
                'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes',
                'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance',
                'histogram_tendency'
            ]

            feature_higher_descriptions = [
                "baseline_value was higher than normal. A higher baseline heart rate may indicate fetal tachycardia, which can be a sign of distress.",
                "accelerations were higher than normal. Increased accelerations can indicate a healthy fetal response to stimuli, but excessive accelerations may also suggest fetal stress.",
                "fetal_movement was higher than normal. Increased fetal movement can be a sign of fetal well-being, but excessive movement may also indicate fetal distress.",
                "uterine_contractions were higher than normal. Increased contractions can indicate uterine irritability or potential preterm labor, which may require monitoring.",
                "light_decelerations were higher than normal. Increased light decelerations could signal umbilical cord compression or placental insufficiency.",
                "severe_decelerations were higher than normal. Any increase in severe decelerations is highly concerning and often indicates acute fetal distress.",
                "prolongued_decelerations were higher than normal. Any amount of prolonged decelerations can be dangerous, as they suggest the fetus is not recovering well from contractions.",
                "abnormal_short_term_variability was higher than normal. Higher abnormal variability in heart rate may point to hypoxia, the deprivation of oxygen to the fetus.",
                "mean_value_of_short_term_variability was higher than normal. A higher mean value might indicate hypoxia, the deprivation of oxygen to the fetus.",
                "percentage_of_time_with_abnormal_long_term_variability was higher than normal. A higher percentage suggests the fetus is experiencing prolonged irregular heart rate patterns, potentially due to distress.",
                "mean_value_of_long_term_variability was higher than normal. High long-term variability may reflect unstable autonomic function, especially if inconsistent with other indicators.",
                "histogram_width was higher than normal. A wider histogram indicates more fluctuation in heart rate, which can point to instability or poor regulation.",
                "histogram_min was higher than normal.  lower minimum is typically worse, but a high minimum could mask variability, signaling a loss of normal heart rate fluctuations.",
                "histogram_max was higher than normal.  A high max value could point to abnormal spikes in fetal heart rate that may not be sustainable or healthy.",
                "histogram_number_of_peaks was higher than normal. A higher number of peaks may indicate more variability in heart rate, which can be a sign of fetal distress.",
                "histogram_number_of_zeroes was higher than normal. A higher number of zeroes suggests periods of no heart rate variability, which can indicate fetal distress.",
                "histogram_mode was higher than normal. A higher mode can indicate a consistent elevation in fetal heart rate, which may be a sign of distress.",
                "histogram_mean was higher than normal. A higher mean heart rate can can indicate a consistent elevation in fetal heart rate, which may be a sign of distress.",
                "histogram_median was higher than normal. A higher median heart rate can indicate a consistent elevation in fetal heart rate, which may be a sign of distress.",
                "histogram_variance was higher than normal.  High variance suggests erratic heart rate behavior.",
                "histogram_tendency was higher than normal. A higher tendency indicates a consistent trend in heart rate, which can suggest a shift in fetal condition."
            ]

            feature_lower_descriptions = [
                "baseline_value was lower than normal. A lower baseline heart rate may indicate fetal bradycardia, which could signal a heart problem or hypoxia, the deprivation of oxygen to the fetus.",
                "accelerations were lower than normal. Reduced accelerations might reflect a lack of fetal responsiveness or fetal hypoxia, the deprivation of oxygen to the fetus.",
                "fetal_movement was lower than normal. Decreased fetal movement can be a sign of fetal sleep cycles but may also indicate fetal compromise.",
                "uterine_contractions were lower than normal. Fewer contractions might reduce stress on the fetus, but may also suggest labor is not progressing.",
                "light_decelerations were lower than normal. Fewer light decelerations are typically a good sign, but an absence of variability might still be concerning.",
                "severe_decelerations were lower than normal. This is generally a positive finding, as fewer severe decelerations suggest less fetal stress.",
                "prolongued_decelerations were lower than normal. Fewer prolonged decelerations are a good sign and suggest better fetal recovery.",
                "abnormal_short_term_variability was lower than normal. Less abnormal variability generally indicates more stable fetal heart rate patterns.",
                "mean_value_of_short_term_variability was lower than normal. A low short-term variability may suggest suppressed nervous system activity or hypoxia, the deprivation of oxygen to the fetus.",
                "percentage_of_time_with_abnormal_long_term_variability was lower than normal. A lower percentage is typically a good indicator of stable fetal health.",
                "mean_value_of_long_term_variability was lower than normal. Reduced long-term variability may reflect poor autonomic control and can be a sign of fetal compromise.",
                "histogram_width was lower than normal. A narrower histogram may suggest reduced variability, potentially indicating distress or poor regulation.",
                "histogram_min was lower than normal. A lower minimum may reflect concerning dips in fetal heart rate, especially if sustained.",
                "histogram_max was lower than normal. A lower max value might indicate less variability or limited fetal response, which could be problematic.",
                "histogram_number_of_peaks was lower than normal. Fewer peaks may indicate reduced variability, potentially associated with fetal compromise.",
                "histogram_number_of_zeroes was lower than normal. Fewer zeroes may suggest some heart rate variability is present, which is usually a good sign.",
                "histogram_mode was lower than normal. A low mode might indicate the fetus is spending more time at a lower heart rate, which could be normal or worrisome depending on other factors.",
                "histogram_mean was lower than normal. A lower mean heart rate might reflect bradycardia or low fetal activity.",
                "histogram_median was lower than normal. A low median could indicate an overall suppressed heart rate, which may be a sign of distress.",
                "histogram_variance was lower than normal. Low variance reflects limited fluctuation in heart rate, possibly signaling compromised fetal autonomic regulation.",
                "histogram_tendency was lower than normal. A negative or flat tendency suggests the heart rate is not trending upward, which might indicate lack of fetal response or deteriorating condition."
            ]
            feature_descriptions = dict(zip(feature_names, zip(feature_higher_descriptions, feature_lower_descriptions)))

            healthy_average_values = [
                133.3, 0.003, 0.009, 0.004, 0.002, 0.0, 0.0, 0.47, 1.3, 9.8, 8.1, 70.4, 93.5,
                164, 4, 0.32, 137.45, 134.61, 138.09, 18.8, 0.32
            ]
            healthy_averages = dict(zip(feature_names, healthy_average_values))

            data = np.array([inputs])
            data_scaled = sc.transform(data)
            
            prediction = model.predict(data_scaled)[0]
            probabilities = model.predict_proba(data_scaled)[0]
            confidence = round(float(probabilities[prediction]) * 100, 2)

            shap_vals = explainer(data_scaled)
            shap_values = shap_vals.values[0, :, prediction]

            top_indices = np.argsort(np.abs(shap_values))[::-1][:3]

            important_features = [{
                "name": feature_names[i],
                "shap": round(float(shap_values[i]), 4),
                "value": round(float(data[0][i]), 4)
            } for i in top_indices]

            return render_template('index.html', prediction=prediction, confidence=confidence, important_features=important_features, healthy_averages=healthy_averages, feature_descriptions=feature_descriptions)

        except Exception as e:
            print("Exception during prediction:", e)
            return render_template('index.html', prediction="Error", confidence="Error")

    return render_template('index.html', prediction=None, confidence=None)

@app.route('/notebook')
def notebook():
    return render_template('notebook.html')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
