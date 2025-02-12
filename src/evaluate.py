from model import load_model
from sklearn.metrics import root_mean_squared_error
import pandas as pd
from scipy import stats as st
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def calc_ci(y_test,final_y_predictions):
  sqr_errors = (final_y_predictions - y_test.to_numpy().ravel()) ** 2
  ci = np.sqrt(st.t.interval(confidence=0.95,df=len(sqr_errors)-1,loc=np.mean(sqr_errors),scale=(st.sem(sqr_errors))))
  return ci


def test_model(path_to_model):
  #load test set from disk
  X_test = pd.read_csv("data/test/X_test.csv")
  y_test = pd.read_csv("data/test/y_test.csv")
  model = load_model(path_to_model)
  y_predictions = model.predict(X_test)
  rmse = root_mean_squared_error(y_test,y_predictions)
  ci = calc_ci(y_test,y_predictions)
  logging.info(f"RMSE on test set: {rmse}")
  logging.info(f"Confidence interval for the RMSE: {ci}")
  return rmse,ci

if __name__ == "__main__":
  test_model("models/best_model.pkl")