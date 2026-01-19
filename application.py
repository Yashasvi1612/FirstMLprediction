from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load trained model and scaler
ridge_model = pickle.load(open('ridge.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method=='POST':
      Temperature = float(request.form['Temperature'])
      RH = float(request.form['RH'])
      Ws = float(request.form['Ws'])
      Rain = float(request.form['Rain'])
      FFMC= float(request.form['FFMC'])
      DMC= float(request.form['DMC'])
      ISI= float(request.form['ISI'])
      Classes= float(request.form['Classes'])
      Region= int(request.form['Region'])  


      new_scaled_data=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])    
      result=ridge_model.predict(new_scaled_data)
      return render_template('home.html', results=result[0])
         
    else:
      return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
