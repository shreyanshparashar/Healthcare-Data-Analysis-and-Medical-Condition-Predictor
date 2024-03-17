from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        name=request.form['name']
        age = int(request.form['age'])
        blood_group = request.form['blood_group']
        gender = request.form['gender']
       
        # Map blood group and gender to one-hot encoded format
        blood_group_mapping = {'A+': [1, 0, 0, 0, 0, 0, 0, 0, 0],
                               'A-': [0, 1, 0, 0, 0, 0, 0, 0, 0],
                               'AB+': [0, 0, 1, 0, 0, 0, 0, 0, 0],
                               'AB-': [0, 0, 0, 1, 0, 0, 0, 0, 0],
                               'B+': [0, 0, 0, 0, 1, 0, 0, 0, 0],
                               'B-': [0, 0, 0, 0, 0, 1, 0, 0, 0],
                               'O+': [0, 0, 0, 0, 0, 0, 1, 0, 0],
                               'O-': [0, 0, 0, 0, 0, 0, 0, 1, 0]}
       
        gender_mapping = {'Male': [1, 0],
                          'Female': [0, 1]}
       
        blood_group_encoded = blood_group_mapping.get(blood_group, [0] * 9)  # 9 features for blood group
        gender_encoded = gender_mapping.get(gender, [0, 0])
       
        # Create input data as a DataFrame
        input_data = pd.DataFrame({'Age': [age],
                                   'Blood Group Type_A+': blood_group_encoded[0],
                                   'Blood Group Type_A-': blood_group_encoded[1],
                                   'Blood Group Type_AB+': blood_group_encoded[2],
                                   'Blood Group Type_AB-': blood_group_encoded[3],
                                   'Blood Group Type_B+': blood_group_encoded[4],
                                   'Blood Group Type_B-': blood_group_encoded[5],
                                   'Blood Group Type_O+': blood_group_encoded[6],
                                   'Blood Group Type_O-': blood_group_encoded[7],
                                   'Gender_Male': gender_encoded[0],
                                   'Gender_Female': gender_encoded[1]})
       
        # Get the feature names used during model training
        feature_names = model.feature_names_in_
       
        # Reorder the input data columns to match the order of feature names
        input_data = input_data[feature_names]
       
        # Predict using the model
        prediction = model.predict(input_data)
       
        # Get the predicted medical condition
        predicted_condition = prediction[0]
        user_data = pd.DataFrame({'Name': [name], 'Age': [age], 'Blood Group': [blood_group], 'Gender': [gender], 'Predicted Condition': [predicted_condition]})
        user_data.to_csv('user_data.csv', mode='a', header=False, index=False)
        print(predicted_condition,"this a output")
       
        return render_template('result.html', predicted_condition=predicted_condition)


if __name__ == '__main__':
    app.run(debug=True)
