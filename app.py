from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

model = joblib.load("9_clusters.pkl")
scaler = joblib.load("scaler.pkl")

roles = {
    0: "Координатор",
    1: "Аналитик",
    2: "Генератор идей",
    3: "Вдохновитель",
    4: "Специалист",
    5: "Командный работник",
    6: "Формирователь",
    7: "Исследователь ресурсов",
    8: "Доводчик",
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if "csvfile" in request.files:
            file = request.files["csvfile"]
            if file.filename.endswith(".csv"):
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)
                try:
                    df = pd.read_csv(filepath, sep="\t", index_col=0)
                    print(df)
                    features = df.select_dtypes(include=["float64", "int64"])
                    scaled = scaler.transform(features)
                    clusters = model.predict(scaled)
                    df["role"] = [roles.get(c, "Неопределённая роль") for c in clusters]
                    print(df)
                    result_path = os.path.join(
                        app.config["UPLOAD_FOLDER"], "result.csv"
                    )
                    df.to_csv(result_path, index=False, encoding="utf-8-sig")
                    return send_file(result_path, as_attachment=True)
                except Exception as e:
                    prediction = f"Ошибка обработки файла: {str(e)}"
                finally:
                    print("here")
            else:
                prediction = "Загрузите CSV-файл с данными."
            print("here")
    return render_template("index.html", prediction=prediction)


def load_features():
    with open("features.txt", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


@app.route("/manual", methods=["GET", "POST"])
def manual():
    prediction = None
    input_values = None
    features = load_features()
    if request.method == "POST":
        try:
            column_names = """  EXT1    EXT2	EXT3    EXT4	EXT5	
                                EXT6	EXT7	EXT8    EXT9	EXT10   
                            	EST1	EST2	EST3	EST4	EST5	
                                EST6	EST7	EST8	EST9	EST10	
                                AGR1	AGR2	AGR3	AGR4	AGR5	
                                AGR6	AGR7	AGR8	AGR9	AGR10	
                                CSN1	CSN2	CSN3	CSN4	CSN5	
                                CSN6	CSN7	CSN8	CSN9	CSN10	
                                OPN1	OPN2	OPN3	OPN4	OPN5	
                                OPN6	OPN7	OPN8	OPN9	OPN10"""
            column_names = column_names.split()

            input_values = [float(request.form[f"feature{i}"]) for i in range(1, 51)]

            input_df = pd.DataFrame([input_values], columns=column_names)

            scaled_input = scaler.transform(input_df)

            cluster = model.predict(scaled_input)[0]
            prediction = roles.get(cluster, "Неопределённая роль")
        except Exception as e:
            prediction = f"Ошибка: {str(e)}"
    return render_template(
        "manual.html", prediction=prediction, values=input_values, features=features
    )


@app.route("/train", methods=["GET", "POST"])
def train():
    message = None
    if request.method == "POST":
        if "csvfile" in request.files:
            file = request.files["csvfile"]
            model_filename = request.form.get("model_filename", "model.pkl")
            scaler_filename = request.form.get("scaler_filename", "scaler.pkl")

            if not model_filename.endswith(".pkl"):
                model_filename += ".pkl"
            if not scaler_filename.endswith(".pkl"):
                scaler_filename += ".pkl"

            if file.filename.endswith(".csv"):
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)
                try:
                    data = pd.read_csv(filepath, sep="\t")
                    data = data.drop(
                        [
                            "dateload",
                            "screenw",
                            "screenh",
                            "introelapse",
                            "testelapse",
                            "endelapse",
                            "IPC",
                            "country",
                            "lat_appx_lots_of_err",
                            "long_appx_lots_of_err",
                        ],
                        axis=1,
                        errors="ignore",
                    )
                    data = data[data.columns.drop(list(data.filter(regex=".*_E")))]
                    data = data[data > 0].dropna()

                    features = data.select_dtypes(include=["float64", "int64"])
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(features)

                    model = KMeans(n_clusters=9, random_state=42)
                    model.fit(scaled_features)

                    joblib.dump(model, model_filename)
                    joblib.dump(scaler, scaler_filename)

                    message = f'Модель сохранена в "{model_filename}", масштабировщик — в "{scaler_filename}".'
                except Exception as e:
                    message = f"Ошибка при обучении: {str(e)}"
            else:
                message = "Пожалуйста, загрузите CSV-файл."
    return render_template("train.html", message=message)


@app.route("/description")
def roles_page():
    return render_template("description.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5100, debug=True)
