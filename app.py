from flask import Flask, render_template, request
from xgboost import XGBRegressor
import pandas as pd

app = Flask(__name__)

@app.route("/")
def FUN_root():
    return render_template("UI.html")

@app.route('/', methods =["GET", "POST"])
def prediction():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       gender = request.form.get("gender")
       age = request.form.get("age")
       h = request.form.get("height")
       w = request.form.get("weight")
       hr = request.form.get("hrate")
       dur = request.form.get("duration")
       tp = request.form.get("temp")

       df = pd.DataFrame([[int(gender), int(age),float(h), float(w), float(dur), float(hr), float(tp)]])
       df = pd.DataFrame(data=df.values, columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp'])
       result = getCaloryPrediction(df)

    return "<h1><center>"+str(result[0])+" kcal""</center></h1>"


def getCaloryPrediction(df):
    m = XGBRegressor()
    m.load_model("./m1.model")

    return m.predict(df)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')