import pip
import torch

import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open(r'C:\Users\kurapatb\Desktop\Sentiment Analysis\sentiment_model_vsa.pkl','rb'))

@app.route('/sentiment_api/',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using model loaded from disk as per the data.
    prediction = model(data['text'])
    # prediction = model.predict([[np.array(data['text'])]])

    # Take the first value of prediction
    '''output = prediction[0]['label']
    output = prediction[0]['score']'''
    l=prediction[0]['label']
    if l=='LABEL_2':
        prediction[0]['label']="positive"
    elif l=="LABEL_1":
        prediction[0]['label']="neutral"
    else:
        prediction[0]['label']="negative"
    output = prediction[0]


    return jsonify(output)

if __name__ == '__main__':
    try:
        app.run()
        #app.run(port=5040, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")