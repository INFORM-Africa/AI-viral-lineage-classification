import argparse
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import re
from utils import load_data, predict_and_save
from joblib import load
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser(description="Random Forest")
    parser.add_argument('-v', '--Data', choices=['SARS-CoV-2', 'HIV'], required=True,
                        help='Specify the virus dataset.')
    parser.add_argument('-f', '--Feature_Description', type=str, required=True,
                        help='Describe the feature set.')
    args = parser.parse_args()


    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.Data)
    
    print(f'Running Random Forest Predictions for {args.Feature_Description}...')
    model_name = 'random_forest'
    model = load(f'models/active/{args.Data}/random_forest.joblib')
    predict_and_save(model, X_train, y_train, args.Feature_Description, "training", args.Data, model_name)
    predict_and_save(model, X_val, y_val, args.Feature_Description, "validation", args.Data, model_name)
    predict_and_save(model, X_test, y_test, args.Feature_Description, "testing", args.Data, model_name)
    print(f"Random Forest Predictions for '{args.Feature_Description}' with original targets saved.")
    
    
    print(f'Running XGBoost Predictions for {args.Feature_Description}...')
    model_name = 'xgboost'
    xgb_clf = XGBClassifier()
    xgb_clf.load_model(f'models/active/{args.Data}/xgboost.json')
    predict_and_save(model, X_train, y_train, args.Feature_Description, "training", args.Data, model_name)
    predict_and_save(model, X_val, y_val, args.Feature_Description, "validation", args.Data, model_name)
    predict_and_save(model, X_test, y_test, args.Feature_Description, "testing", args.Data, model_name)
    print(f"XGBoost Predictions for '{args.Feature_Description}' with original targets saved.")
    
    
    print(f'Running Random Forest Predictions for {args.Feature_Description}...')
    model_name = 'knn'
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    model = load(f'models/active/{args.Data}/knn.joblib')
    predict_and_save(model, X_train, y_train, args.Feature_Description, "training", args.Data, model_name)
    predict_and_save(model, X_val, y_val, args.Feature_Description, "validation", args.Data, model_name)
    predict_and_save(model, X_test, y_test, args.Feature_Description, "testing", args.Data, model_name)
    print(f"Predictions for '{args.Feature_Description}' with original targets saved.")

if __name__ == "__main__":
    main()