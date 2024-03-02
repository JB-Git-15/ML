import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

#model = pickle.load(open('log_reg.pkl', 'rb'))


@app.route('/')
def rookie_prediction():
    return render_template('./templates/rook.html')

"""
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    features = [np.array(int_features)]  
    prediction = model.predict(features) 
    result = prediction[0]

    return render_template('Rookie_prediction.html', prediction=result)
"""

if __name__ == "__main__":
    app.run(debug=True)
