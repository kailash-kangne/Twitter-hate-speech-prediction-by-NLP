from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np

app = Flask(__name__)

clf=pickle.load(open('clf.pkl','rb'))
cv=pickle.load(open('cv.pkl','rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        t = str(request.form['tweet'])
        text = cv.transform([t]).toarray()
        output = clf.predict(text)
        return render_template('index.html', prediction_text="{}".format(output))

if __name__=="__main__":
    app.run(debug=True)