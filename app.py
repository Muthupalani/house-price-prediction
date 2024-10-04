from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('house_price_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form input data
    area = int(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    mainroad = int(request.form['mainroad'])
    guestroom = int(request.form['guestroom'])
    basement = int(request.form['basement'])
    hotwaterheating = int(request.form['hotwaterheating'])
    airconditioning = int(request.form['airconditioning'])
    parking = int(request.form['parking'])
    prefarea = int(request.form['prefarea'])
    furnishingstatus = int(request.form['furnishingstatus'])

    # Create a numpy array with inputs
    input_features = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]])

    # Prediction
    prediction = model.predict(input_features)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
