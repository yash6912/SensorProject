
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
    make_response
)

from src.exception import CustomException
from src.logger import logging as lg
from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

import sys

app = Flask(__name__)


@app.route("/")
def home():
    return redirect(url_for("upload"))


@app.route("/train")
def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()

        return "Training Completed."

    except Exception as e:
        raise CustomException(e, sys)


@app.route("/predict", methods=["GET", "POST"])
def upload():
    try:
        if request.method == "POST":

            uploaded_file = request.files.get("file")

            # Validate file selection
            if not uploaded_file or not uploaded_file.filename:
                return render_template(
                    "upload_file.html",
                    error="Please select a CSV file before running a prediction."
                ), 400

            # Validate file extension
            if not uploaded_file.filename.lower().endswith(".csv"):
                return render_template(
                    "upload_file.html",
                    error="Invalid file format. Please upload a CSV file."
                ), 400

            # Run the existing prediction pipeline
            prediction_pipeline = PredictionPipeline(request)

            prediction_file_detail = prediction_pipeline.run_pipeline()

            lg.info("Prediction completed. Preparing CSV download.")

            # Prepare the downloadable prediction file
            response = make_response(
                send_file(
                    prediction_file_detail.prediction_file_path,
                    download_name=prediction_file_detail.prediction_file_name,
                    as_attachment=True,
                    mimetype="text/csv"
                )
            )

            # Notify the browser that Flask returned the download response
            download_token = request.form.get("download_token")

            if download_token:
                response.set_cookie(
                    "download_token",
                    download_token,
                    max_age=120,
                    path="/",
                    samesite="Lax"
                )

            return response

        return render_template("upload_file.html")

    except Exception as e:
        lg.exception("Error during wafer prediction")
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
