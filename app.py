from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and prepare the dataset
def load_data():
    # Load the dataset with the correct column names
    data = pd.read_csv('housing_data.csv')  # Assuming the CSV has headers

    # Print the first few rows of the dataset
    print("Dataset Preview:")
    print(data.head())

    # Check the data types of each column
    print("\nData Types:")
    print(data.dtypes)

    # Convert relevant columns to numeric, coercing errors to NaN
    data['Bedrooms'] = pd.to_numeric(data['Bedrooms'], errors='coerce')
    data['Bathrooms'] = pd.to_numeric(data['Bathrooms'], errors='coerce')
    data['SquareFeet'] = pd.to_numeric(data['SquareFeet'], errors='coerce')
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')  # Ensure price is numeric

    # Drop rows with NaN values
    data = data.dropna()

    # One-hot encode the 'ZipCode' column
    data = pd.get_dummies(data, columns=['ZipCode'], drop_first=True)

    # Store the list of one-hot encoded columns
    global list_of_zipcode_columns
    list_of_zipcode_columns = [col for col in data.columns if col.startswith('ZipCode_')]

    # Separate features and target variable
    X = data.drop('Price', axis=1)  # Features
    y = data['Price']  # Target variable
    return X, y

# Train the model
def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, scaler

model, scaler = train_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect form data
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        square_feet = float(request.form['size'])
        zip_code = request.form['location']  # This should match the categories used in training

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'SquareFeet': [square_feet],
            'YearBuilt': [0],  # Default value
            'GarageSpaces': [0],  # Default value
            'LotSize': [0],  # Default value
            'ZipCode': [zip_code],  # This will be one of the categories
            'CrimeRate': [0],  # Default value
            'SchoolRating': [0]  # Default value
        })

        # One-hot encode the ZipCode just like you did during training
        input_data = pd.get_dummies(input_data, columns=['ZipCode'], drop_first=True)

        # Define expected columns based on your training data
        expected_columns = [
            'Bedrooms', 'Bathrooms', 'SquareFeet', 'YearBuilt', 'GarageSpaces', 
            'LotSize', 'CrimeRate', 'SchoolRating'
        ] + list_of_zipcode_columns  # Add your one-hot encoded ZipCode columns here

        # Fill missing columns with 0
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0  # Fill missing columns with 0

        # Reorder columns to match the training data
        input_data = input_data[expected_columns]

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        return render_template('index.html', prediction=prediction[0])

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
