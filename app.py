from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

modell = pickle.load(open("model.pkl", "rb"))
dataset = pd.read_csv('diabetes.csv')
dataset_X = dataset.iloc[:,[0,1,2,3,4,5,6,7]].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    text7 = request.form['7']
    text8 = request.form['8']
 
    row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8])])
    print(row_df)
    final = row_df.astype(float)
    prediction=modell.predict_proba(sc.transform(final))
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('index.html',pred='You might have chance of having diabetes.')
    else:
        return render_template('index.html',pred='You are safe.')





if __name__ == '__main__':
    app.run(debug=True)
