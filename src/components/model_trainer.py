
import os
import sys
import pickle
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.constant import *
from src.exception import CustomException
from src.utils.main_utils import MainUtils


@dataclass
class ModelTrainerConfig:
    artifact_folder: str = os.path.join(artifact_folder)
    train_model_path: str = os.path.join(
        artifact_folder, "model.pkl"
    )
    expected_accuracy: float = 0.45
    model_config_file_path: str = os.path.join(
        "config", "model.yaml"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        self.models = {
            "XGBClassifier": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
            ),
            "GradientBoostingClassifier":
                GradientBoostingClassifier(random_state=42),
            "SVC": SVC(),
            "RandomForestClassifier":
                RandomForestClassifier(random_state=42),
        }

    def evaluate_models(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        models,
    ):
        """Train and evaluate all candidate models."""

        try:
            report = {}

            for model_name, model in models.items():
                logging.info(
                    "Training model: %s", model_name
                )

                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_score = accuracy_score(
                    y_train, y_train_pred
                )
                test_score = accuracy_score(
                    y_test, y_test_pred
                )

                report[model_name] = {
                    "train_accuracy": train_score,
                    "test_accuracy": test_score,
                }

                logging.info(
                    "%s | Train accuracy: %.4f | "
                    "Test accuracy: %.4f",
                    model_name,
                    train_score,
                    test_score,
                )

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model(self, model_report):
        """Return the model with the highest test accuracy."""

        try:
            if not model_report:
                raise ValueError(
                    "Model evaluation report is empty."
                )

            best_model_name = max(
                model_report,
                key=lambda name: model_report[name][
                    "test_accuracy"
                ],
            )

            best_model_score = model_report[
                best_model_name
            ]["test_accuracy"]

            best_model_object = self.models[
                best_model_name
            ]

            logging.info(
                "Best model: %s | Accuracy: %.4f",
                best_model_name,
                best_model_score,
            )

            return (
                best_model_name,
                best_model_object,
                best_model_score,
            )

        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(
        self,
        best_model_object,
        best_model_name,
        X_train,
        y_train,
    ):
        """Tune the selected model using GridSearchCV."""

        try:
            logging.info(
                "Loading model configuration from: %s",
                self.model_trainer_config
                .model_config_file_path,
            )

            config = self.utils.read_yaml_file(
                self.model_trainer_config
                .model_config_file_path
            )

            model_config = config.get("model", {})

            # Support either a model-specific YAML section
            # or a direct GridSearchCV parameter dictionary.
            if best_model_name in model_config:
                param_grid = model_config[
                    best_model_name
                ]
            else:
                param_grid = model_config

            if not param_grid:
                logging.warning(
                    "No hyperparameters configured for %s. "
                    "Returning the original trained model.",
                    best_model_name,
                )
                return best_model_object

            grid_search = GridSearchCV(
                estimator=best_model_object,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1,
                scoring="accuracy",
                error_score="raise",
            )

            grid_search.fit(X_train, y_train)

            logging.info(
                "Best parameters: %s",
                grid_search.best_params_,
            )

            logging.info(
                "Best cross-validation accuracy: %.4f",
                grid_search.best_score_,
            )

            return grid_search.best_estimator_

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(
        self,
        train_array,
        test_array,
    ):
        """Train, select, tune, evaluate, and save the model."""

        try:
            logging.info(
                "Starting model training."
            )

            # Last column is assumed to be the target.
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info(
                "Training data shape: %s",
                X_train.shape,
            )
            logging.info(
                "Testing data shape: %s",
                X_test.shape,
            )

            # Step 1: Evaluate candidate models.
            model_report = self.evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=self.models,
            )

            print("\nModel Evaluation Report:")
            for name, scores in model_report.items():
                print(
                    f"{name}: "
                    f"Train Accuracy = "
                    f"{scores['train_accuracy']:.4f}, "
                    f"Test Accuracy = "
                    f"{scores['test_accuracy']:.4f}"
                )

            # Step 2: Select the best model.
            (
                best_model_name,
                best_model_object,
                best_model_score,
            ) = self.get_best_model(model_report)

            print(
                f"\nBest model before tuning: "
                f"{best_model_name}"
            )
            print(
                f"Test accuracy before tuning: "
                f"{best_model_score:.4f}"
            )

            # Step 3: Tune the selected model.
            best_model_object = self.finetune_best_model(
                best_model_object=best_model_object,
                best_model_name=best_model_name,
                X_train=X_train,
                y_train=y_train,
            )

            # Step 4: Evaluate the tuned model on held-out data.
            y_test_pred = best_model_object.predict(X_test)

            final_accuracy = accuracy_score(
                y_test, y_test_pred
            )

            logging.info(
                "Final test accuracy: %.4f",
                final_accuracy,
            )

            print(
                f"Final test accuracy after tuning: "
                f"{final_accuracy:.4f}"
            )

            # Step 5: Check the expected accuracy threshold.
            if (
                final_accuracy
                < self.model_trainer_config.expected_accuracy
            ):
                raise ValueError(
                    "No acceptable model found. "
                    f"Expected accuracy: "
                    f"{self.model_trainer_config.expected_accuracy:.4f}, "
                    f"obtained accuracy: {final_accuracy:.4f}"
                )

            # Step 6: Save the trained model.
            model_path = (
                self.model_trainer_config.train_model_path
            )

            os.makedirs(
                os.path.dirname(model_path) or ".",
                exist_ok=True,
            )

            with open(model_path, "wb") as file:
                pickle.dump(best_model_object, file)

            logging.info(
                "Trained model saved at: %s",
                model_path,
            )

            print(f"Model saved successfully at: {model_path}")

            return {
                "model_name": best_model_name,
                "test_accuracy": final_accuracy,
                "model_path": model_path,
            }

        except Exception as e:
            raise CustomException(e, sys)
