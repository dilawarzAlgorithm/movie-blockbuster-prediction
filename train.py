import os

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Configuration
DATA_PATH = './data/movie_blockbusters.csv'
MODEL_OUTPUT_FILE = './model/model.bin'
ETA = 0.1
MAX_DEPTH = 3
MIN_CHILD_WEIGHT = 1
GAMMA = 7.0

os.makedirs(os.path.dirname(MODEL_OUTPUT_FILE), exist_ok=True)

def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()

    # Clean decade column
    if df['decade'].dtype == 'string':
        df['decade'] = df['decade'].str.removesuffix('s').astype(int)

    # Remove post-release metrics to prevent data leakage
    post_release_cols = [
        'movie_id', 'title', 'revenue_usd', 'profit_usd', 
        'roi', 'rating', 'vote_count', 'popularity'
    ]
    df = df.drop(columns=[col for col in post_release_cols if col in df.columns])
    
    return df

def prepare_features(df_input):
    df_out = df_input.copy()
    
    # 1. Log-transform skewed budget
    df_out['log_budget'] = np.log1p(df_out['budget_usd'])
    
    # 2. Financial intensity ratio (budget per minute)
    df_out['budget_per_min'] = df_out['budget_usd'] / df_out['runtime_min']
    
    return df_out

def train_and_evaluate():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Partitioning dataset into full_train and test splits...")
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df.blockbuster, random_state=42
    )

    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train.blockbuster.values
    y_test = df_test.blockbuster.values

    del df_full_train['blockbuster']
    del df_test['blockbuster']

    print("Applying feature engineering...")
    df_full_train_fe = prepare_features(df_full_train)
    df_test_fe = prepare_features(df_test)

    print("Encoding features with DictVectorizer...")
    full_train_dicts = df_full_train_fe.to_dict(orient='records')
    test_dicts = df_test_fe.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(full_train_dicts)
    X_test = dv.transform(test_dicts)

    features = list(dv.get_feature_names_out())
    d_full_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
    d_test = xgb.DMatrix(X_test, label=y_test, feature_names=features)

    print("Training final XGBoost model...")
    scale_pos_weight = (len(y_full_train) - sum(y_full_train)) / sum(y_full_train)

    xgb_params = {
        'eta': ETA,
        'max_depth': MAX_DEPTH,
        'min_child_weight': MIN_CHILD_WEIGHT,
        'gamma': GAMMA,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.train(xgb_params, d_full_train, num_boost_round=175)

    print("Evaluating model performance on test set...")
    y_pred = model.predict(d_test)
    auc = roc_auc_score(y_test, y_pred)
    print(f"Final Test ROC-AUC: {auc:.4f}")

    print(f"Saving model artifacts to {MODEL_OUTPUT_FILE}...")
    with open(MODEL_OUTPUT_FILE, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    
    print("Training completed successfully!")

if __name__ == '__main__':
    train_and_evaluate()