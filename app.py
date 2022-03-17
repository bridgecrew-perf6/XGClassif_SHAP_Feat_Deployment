import werkzeug
import numpy as np
import pandas as pd
from joblib import load
from flask import Flask, request, jsonify, render_template

# Creating app object

app = Flask(__name__)

# Predicting Test data
# Checking the model with single data point from test dataset
# Loading required models

# Loading the saved model
full_model_pipeline = load('final_model_pipeline')

# Using the html template
@app.route('/')
def home():
    return render_template('index.html')

# Exposing the below code to localhost:5000
@app.route('/api', methods=['POST'])
def pred_testquery():
#xgb_features = ['marital-status_ Married-civ-spouse', 'capital_loss','education_num', 'occupation_ Exec-managerial',
                # 'capital_gain', 'age', 'hours_per_week', 'sex_ Female', 'relationship_ Own-child',
                # 'occupation_ Other-service', 'occupation_ Prof-specialty', 'fnlwgt']
    # content = request.json
    marital_status_Married_civ_spouse = float(request.form['marital-status_ Married-civ-spouse'])
    capital_loss = float(request.form['capital_loss'])
    education_num = float(request.form['education_num'])
    occupation_Exec_managerial = float(request.form['occupation_ Exec-managerial'])
    capital_gain = float(request.form['capital_gain'])
    age = float(request.form['age'])
    hours_per_week = float(request.form['hours_per_week'])
    Sex_Female = float(request.form['sex_ Female'])
    relationship_Own_child = float(request.form['relationship_ Own-child'])
    occupation_Other_service = float(request.form['occupation_ Other-service'])
    occupation_Prof_specialty = float(request.form['occupation_ Prof-specialty'])
    fnlwgt = float(request.form['fnlwgt'])

    datapoint = [marital_status_Married_civ_spouse, capital_loss, education_num, occupation_Exec_managerial,
                 capital_gain, age, hours_per_week, Sex_Female, relationship_Own_child, occupation_Other_service,
                 occupation_Prof_specialty, fnlwgt]

    def y_pred(X_data=datapoint, model=full_model_pipeline):

        # Creating a dataframe out of input
        df = pd.DataFrame(index=[0])
        df['marital_status_Married_civ_spouse'] = X_data[0]
        df['capital_loss'] = X_data[1]
        df['education_num'] = X_data[2]
        df['occupation_Exec_managerial'] = X_data[3]
        df['capital_gain'] = X_data[4]
        df['age'] = X_data[5]
        df['hours_per_week'] = X_data[6]
        df['Sex_Female'] = X_data[7]
        df['relationship_Own_child'] = X_data[8]
        df['occupation_Other_service'] = X_data[9]
        df['occupation_Prof_specialty'] = X_data[10]
        df['fnlwgt'] = X_data[11]

        # Fitting classifier model
        pred = model.named_steps.xgb.predict(df)

        return pred

    prediction = y_pred()
    if prediction == 0:
        above_50K = "Yes"
    else:
        above_50K = "No"
    response = {'Is income of this individual above 50K?': above_50K}

    return str(response)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)