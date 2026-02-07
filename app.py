from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
with open("diabetes-prediction(rfc).pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["bloodpressure"]),
            float(request.form["skinthickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"])
        ]

        final_input = np.array(data).reshape(1, -1)
        prediction = model.predict(final_input)[0]

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return render_template("result.html", prediction=result)

    except:
        return render_template("result.html", prediction="Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)
