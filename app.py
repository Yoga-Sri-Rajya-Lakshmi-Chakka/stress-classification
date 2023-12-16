

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

with open('Stress_detection.pkl', 'rb') as f:
    model = joblib.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        snoring_rate=int(request.form['snoring_rate'])
        respiration_rate=int(request.form['respiration_rate'])
        body_temperature=int(request.form['body_temperature'])
        limb_movement=int(request.form['limb_movement'])
        blood_oxygen=int(request.form['blood_oxygen'])
        eye_movement=int(request.form['eye_movement'])
        sleeping_hours=int(request.form['sleeping_hours'])
        heart_rate=int(request.form['heart_rate'])

        input_data = np.array([[snoring_rate, respiration_rate, body_temperature, limb_movement,blood_oxygen, eye_movement, sleeping_hours, heart_rate]])
        prediction = model.predict(input_data)[0]
        def example(prediction):
            if prediction==0: return 'No Stress'
            elif prediction==1: return 'Low Stress'
            elif prediction==2: return 'Moderate'
            elif prediction==3: return 'High'
            elif prediction==4: return 'Extreme Stress'
            return 'No Stress found!!'
        
        final_answer=example(prediction)

        return render_template('index.html', prediction=final_answer)

if __name__ == '__main__':
           
    app.run(debug=True)
   