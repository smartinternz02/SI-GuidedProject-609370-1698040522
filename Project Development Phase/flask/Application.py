from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np


app = Flask(__name__)

# model = joblib.load('model.pkl')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')  
def home():
    return render_template('home.html')

@app.route('/car_purchase_prediction')  
def car_purchase_prediction():
    result = request.args.get('result', default=None, type=float)
    return render_template('car_purchase_prediction.html', result=result)


@app.route('/educational_content')  
def educational_content():
    return render_template('educational_content.html')

@app.route('/feedback_form')  
def feedback_form():
    return render_template('feedback_form.html')

@app.route('/payment')  
def payment():
    return render_template('payment.html')

@app.route('/reviews')  
def reviews():
    return render_template('reviews.html')


@app.route('/pred', methods=['POST','GET'])
def predict1():
        age = float(request.form["age"])
        income = float(request.form["income"])
        gender=float(request.form["Gender"])
        if request.method == 'POST':
            input_data = [[age, income, gender]]
            prediction = model.predict(input_data)
            result = prediction[0]

        # Redirect to the original page with the result as a query parameter

        return render_template("car_purchase_prediction.html", prediction_text="You are eligible to buy a {}".format(result))


if __name__ == "__main__":
    app.run()
