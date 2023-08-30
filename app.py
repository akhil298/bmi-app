from flask import Flask,render_template,session,url_for,redirect
from wtforms import StringField,SubmitField
import numpy as np
from flask_wtf import FlaskForm
from sklearn.neighbors import KNeighborsClassifier
import pickle
import joblib

def return_prediction(model,sample_json):
  s_height = sample_json['height']
  s_weight = sample_json['weight']
  s_gender = sample_json['Gender']
  classes = np.array(['Extremly Weak', 'Weak', 'Normal', 'Overweight' ,'Obesity', 'Extremly Obesity'])
  ip = [[s_height,s_weight,s_gender]]
  class_ind = model.predict(ip)[0]
  return classes[class_ind]




app = Flask(__name__)
app.config['SECRET_KEY'] = 'MYkeySECRET'


class models(FlaskForm):
	height        = StringField("height")
	weight        = StringField("Weight")
	Gender_female = StringField("Gender_female")
	submit = SubmitField("predict")


@app.route("/",methods=['GET','POST'])
def index():
	form = models()
	if form.validate_on_submit():
		session['height'] = form.height.data
		session['weight'] = form.weight.data
		session['Gender'] = form.Gender.data

		return redirect(url_for("bmi_pred"))

	return render_template('home.html',form=form)

model = pickle.load(open("knn_model.pkl", "rb"))
@app.route('/prediction')
def bmi_pred():
	content = {}
	content['height'] = float(session['height'])
	content['weight'] = float(session['weight'])
	content['Gender'] = float(session['Gender'])

	results = return_prediction(model,content)
	return render_template('prediction.html',results=results)

if __name__ =='__main__':
	app.run()

