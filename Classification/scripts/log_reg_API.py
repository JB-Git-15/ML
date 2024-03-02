from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import sys
application = Flask(__name__)
@application.route('/prediction', methods=['POST'])


def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=rnd_columns, fill_value=0)
            predict = list(lr.predict(query))
            return jsonify({'prediction': str(predict)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('Failed loading the model')
        

        
if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345
        lr = joblib.load('log_reg.pkl')
        rnd_columns = joblib.load('log_reg_columns.pkl')
        application.run(port=port, debug=True)
