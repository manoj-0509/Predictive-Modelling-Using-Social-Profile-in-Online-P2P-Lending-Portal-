from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model_EMI = joblib.load('model_EMI.pkl')
model_ROI = joblib.load('model_ROI.pkl')
model_ELA = joblib.load('model_ELA.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.form.to_dict()

    # Extract the input features from the data
    LP_CustomerPayments = data['LP_CustomerPayments']
    LoanDateYear = data['LoanDateYear']
    Month = data['Month']
    MaturityDatOriginalYear = data['MaturityDateOriginalYear']
    MaturityDateOriginalMonth = data['MaturityDateOriginalMonth']
    LP_CustomerPrincipalPayments = data['LP_CustomerPrincipalPayments']
    TradesNeverDelinquent = data['TradesNeverDelinquent(Percentage)']
    AvailableBankcardCredit = data['AvailableBankcardCredit']
    TotalInquiries = data['TotalInquiries']
    InterestAmount = data['InterestAmount']
    TotalAmount = data['TotalAmount']
    StatedMonthlyIncome = data['StatedMonthlyIncome']
    DebtToIncomeRatio = data['DebtToIncomeRatio']
    IncomeVerifiable = data['IncomeVerifiable']
    LoanOriginalAmount = data['LoanOriginalAmount']
    BorrowerRate = data['BorrowerRate']
    LoanTenure = data['LoanTenure']

    # Create an input data array
    X_new = [[LP_CustomerPayments, LoanDateYear, Month,  MaturityDatOriginalYear, MaturityDateOriginalMonth, LP_CustomerPrincipalPayments, TradesNeverDelinquent, AvailableBankcardCredit, TotalInquiries, InterestAmount, TotalAmount, StatedMonthlyIncome, DebtToIncomeRatio, IncomeVerifiable, LoanOriginalAmount, BorrowerRate, LoanTenure]]

    # Make predictions for all three target variables
    y_pred_EMI = model_EMI.predict(X_new)
    y_pred_ROI = model_ROI.predict(X_new)
    y_pred_EligibleLoanAmount = model_ELA.predict(X_new)

    # Return predictions for all three target variables
    return render_template('index.html', EMI=y_pred_EMI[0], ROI=y_pred_ROI[0], EligibleLoanAmount=y_pred_EligibleLoanAmount[0])

if __name__ == '__main__':
    app.run(debug=True)
