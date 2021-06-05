from flask import Flask, jsonify,request,render_template
import joblib as jb
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app=Flask(__name__)
model=jb.load(".\models\Boom_bike_lr_model.pkl")
scaling=jb.load(".\models\Boom_bike_scaling.pkl")

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/prediction',methods=['POST'])
def predict():
    if (request.method=='POST'):
        features=[float(x) for x in request.form.values()]
        features=[1]+features+[0]
        final_features=np.array(features).reshape(1,12)
        final_features[0][[9,11,10]]=scaling.transform(final_features[0][[9,11,10]].reshape(1,3))
        final_features=final_features[0,0:11]
        final_features = final_features.reshape(1,11)
        prediction_pred=model.predict(final_features)
        return render_template('index.html',prediction_text='Sales output : {}'.format(int(prediction_pred[0])))
    else:
        return render_template('index.html')

if (__name__)==('__main__'):
    app.run(debug=True)
