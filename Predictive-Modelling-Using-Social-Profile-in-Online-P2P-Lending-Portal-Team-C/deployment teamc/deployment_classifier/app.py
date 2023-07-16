from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
app = Flask(__name__)
# load the model
# model = pickle.load(open('C:\Users\omar_\Downloads\flask_project_rahmaAddCSS\classification_model', 'rb'))
model = joblib.load('modeel_class1.pkl')
@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    BorrowerAPR = float(request.form['BorrowerAPR'])
    CreditScoreRangeLower = float(request.form['CreditScoreRangeLower'])
    CreditScoreRangeUpper = float(request.form['CreditScoreRangeUpper'])
    LP_CustomerPrincipalPayments = float(request.form['LP_CustomerPrincipalPayments'])
    EstimatedReturn = float(request.form['EstimatedReturn'])
    LenderYield = float(request.form['LenderYield'])
    LP_CustomerPayments = float(request.form['LP_CustomerPayments'])
    EstimatedLoss = float(request.form['EstimatedLoss'])
    BorrowerRate = float(request.form['BorrowerRate'])

    print('Input Values:', BorrowerAPR, CreditScoreRangeLower, CreditScoreRangeUpper,LP_CustomerPrincipalPayments, EstimatedReturn, LenderYield,LP_CustomerPayments, EstimatedLoss, BorrowerRate)

    input_data = [[BorrowerAPR, CreditScoreRangeLower, CreditScoreRangeUpper,LP_CustomerPrincipalPayments, EstimatedReturn, LenderYield,LP_CustomerPayments, EstimatedLoss, BorrowerRate]]

    print('Input Data:', input_data)

    result = model.predict(input_data)
    # result = model.predict(input_data)[0]

    print('Prediction Result:', result)

    return render_template('index.html', **locals())



if __name__ == '__main__':
    app.run(debug=True)