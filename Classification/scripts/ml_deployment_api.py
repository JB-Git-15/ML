from flask import Flask, request
import pickle

import numpy as np

lr = pickle.load(open("log_reg.pkl",'rb'))


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def hello_world():
    request_data = request.get_json(force= True)
    age = request_data['age']
    prediction = lr.predict()

    return "The prediciton is {}".format(local_scaler.transform(np.array(age)))



if __name__ == "main":
    app.run(port=800, debug=True)
