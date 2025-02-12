import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from preprocessing import preprocessing, remove_outliers, ClusterSimilarity
from model import save_model
from data_download import load_data

logging.basicConfig(level=logging.INFO)
rf_pipeline = Pipeline(steps=[("preprocessing",preprocessing),
                              ("rf",RandomForestRegressor(random_state=42))
                              ])


def train_model(X_train,y_train,path_to_save_model=None):
  """Train the RandomForest model using RandomizedSearchCV."""
  param_distributions = {  
    'rf__max_features': randint(1, X_train.shape[1]),  
    'rf__min_samples_split': randint(2, 20),  
    'rf__bootstrap': [True, False], 
    'rf__min_samples_leaf': randint(1, 20)    
  }
  search = RandomizedSearchCV(rf_pipeline, param_distributions, scoring='neg_root_mean_squared_error', cv=3,n_jobs=-1,verbose=2)
  search.fit(X_train, y_train.to_numpy().ravel())
  logging.info(f"Best parameters: {search.best_params_}")
  logging.info(f"Best cross-validation score: {search.best_score_}")
  if path_to_save_model:
    save_model(search,path_to_save_model)
  return search.best_estimator_
  
loc = ['longitude','latitude']
def main():
    housing_data = load_data()
    housing_data['ocean_proximity'] = housing_data['ocean_proximity'].astype('category')
    housing_data['income_cat'] = pd.cut(housing_data['median_income'], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])
    
    train_set, test_set = train_test_split(housing_data, random_state=42, stratify=housing_data['income_cat'], test_size=0.2)
    train_set.drop("income_cat", axis=1, inplace=True)
    test_set.drop("income_cat", axis=1, inplace=True)
    
    train_set = remove_outliers(train_set, 'median_house_value')
    X_train, y_train = train_set.drop("median_house_value", axis=1), train_set[["median_house_value"]]
    X_test, y_test = test_set.drop("median_house_value", axis=1), test_set[["median_house_value"]]
    
    # Save X_test and y_test to disk. Testing and evaluation is done in evaluate.py
    similarities = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42, init='k-means++', n_init='auto')
    similarities.fit(X_train[['longitude', 'latitude']].values)
    
    if not os.path.exists("data/objects"):
        os.makedirs("data/objects")
    save_model(similarities, "data/objects/cluster_simil.pkl")
    
    X_train[similarities.get_feature_names_out()] = similarities.transform(X_train[['longitude', 'latitude']].values)
    X_test[similarities.get_feature_names_out()] = similarities.transform(X_test[['longitude', 'latitude']].values)
    X_train.drop(['longitude', 'latitude'], axis=1, inplace=True)
    X_test.drop(['longitude', 'latitude'], axis=1, inplace=True)
    
    if not os.path.exists("data/test"):
        os.makedirs("data/test")
    X_test.to_csv("data/test/X_test.csv", index=False)
    y_test.to_csv("data/test/y_test.csv", index=False)
    
    train_model(X_train, y_train, rf_pipeline, "models/best_model.pkl")
  
  
if __name__ == "__main__":
  main()
  logging.info(f"Model trained successfully and saved to models/best_model.pkl")  