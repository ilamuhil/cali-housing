
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,FunctionTransformer,OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def column_ratio(X):
  return X[:,[0]]/X[:,[1]]
def ratio_name(function_transformer,feature_names_in):
  return ["ratio"]



ratio_pipeline = make_pipeline(SimpleImputer(strategy='median'),FunctionTransformer(column_ratio,feature_names_out=ratio_name))

log_pipeline = make_pipeline(SimpleImputer(strategy='median'),FunctionTransformer(np.log,feature_names_out="one-to-one",inverse_func=np.exp),StandardScaler())

catpipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore'))

num_pipeline = make_pipeline(SimpleImputer(strategy='median'),StandardScaler())

