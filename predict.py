import pickle
import numpy as np
import xgboost as xgb

MODEL_FILE = "./model/model.bin"

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)


def prepare_single_feature(movie: dict) -> dict:
    movie_out = movie.copy()

    budget = float(movie_out.get("budget_usd", 0.0))
    runtime = float(movie_out.get("runtime_min", 1.0))

    movie_out["log_budget"] = float(np.log1p(budget))
    movie_out["budget_per_min"] = float(budget / max(runtime, 1.0))

    return movie_out


def predict_blockbuster(movie: dict) -> dict:
    processed_movie = prepare_single_feature(movie)

    X = dv.transform([processed_movie])

    features = list(dv.get_feature_names_out())
    dmatrix = xgb.DMatrix(X, feature_names=features)

    y_pred_prob = float(model.predict(dmatrix)[0])
    is_blockbuster = bool(y_pred_prob >= 0.5)

    return {
        "blockbuster_probability": round(y_pred_prob, 4),
        "is_blockbuster": is_blockbuster,
    }