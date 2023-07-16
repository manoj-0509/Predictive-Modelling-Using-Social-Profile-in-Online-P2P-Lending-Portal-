from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model_EMI = joblib.load('model_EMI.pkl')
model_ROI = joblib.load('model_ROI.pkl')
model_ELA = joblib.load('model_ELA.pkl')
model_Classifier = joblib.load('model_Classifier.pkl')


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
    
    BorrowerAPR = data["BorrowerAPR"]
    CreditScoreRangeLower = data["CreditScoreRangeLower"]
    CreditScoreRangeUpper = data["CreditScoreRangeUpper"]
    LP_CustomerPrincipalPayments2 = data["LP_CustomerPrincipalPayments2"]
    EstimatedReturn = data["EstimatedReturn"]
    LenderYield = data["LenderYield"]
    LenderYield_1 = data["LenderYield_1"]
    LP_CustomerPayments2 = data["LP_CustomerPayments2"]
    EstimatedLoss = data["EstimatedLoss"]
    BorrowerRate2 = data["BorrowerRate2"]

    # Create an input data array
    X_new_1 = [[LP_CustomerPayments, LoanDateYear, Month,  MaturityDatOriginalYear, MaturityDateOriginalMonth, LP_CustomerPrincipalPayments, TradesNeverDelinquent, AvailableBankcardCredit, TotalInquiries, InterestAmount, TotalAmount, StatedMonthlyIncome, DebtToIncomeRatio, IncomeVerifiable, LoanOriginalAmount, BorrowerRate, LoanTenure]]

    X_new_2 = [[BorrowerAPR, CreditScoreRangeLower, CreditScoreRangeUpper, LP_CustomerPrincipalPayments2, EstimatedReturn, LenderYield, LenderYield_1, LP_CustomerPayments2, EstimatedLoss, BorrowerRate2]]

    # Make predictions for all three target variables
    y_pred_EMI = model_EMI.predict(X_new_1)
    y_pred_ROI = model_ROI.predict(X_new_1)
    y_pred_EligibleLoanAmount = model_ELA.predict(X_new_1)
    y_pred_Classifier = model_Classifier.predict(X_new_2)

    # Return predictions for all three target variables
    return render_template('index.html', EMI=y_pred_EMI[0], ROI=y_pred_ROI[0], EligibleLoanAmount=y_pred_EligibleLoanAmount[0], Classifier=y_pred_Classifier[0])

if __name__ == '__main__':
    app.run(debug=True)
