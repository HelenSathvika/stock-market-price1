import pandas as pd
import numpy as np
import code1
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
exec('code1')
model= load('model.save')

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    print(x_test)
    prediction = model.predict(x_test)
    print(prediction)
    output=prediction[0]
    return render_template('Prediction1.html',output=output)

@app.route('/save_data',methods=['POST'])
def save_data():
   '''
   For saving data
   ''' 
   data=[[x for x in request.form.values()]]
   print(data)
   df=pd.DataFrame(data)
   df.to_csv('datasetcanon.csv',mode='a',index=False,header=False)
   return render_template('Saving.html',output1="Saved")
           

if __name__ == "__main__":
    app.run(host='127.0.0.1',port='5000')