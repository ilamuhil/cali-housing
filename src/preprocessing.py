import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from pipeline import ratio_pipeline,log_pipeline,catpipeline,num_pipeline


def remove_outliers(data, column):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if column not in data.columns:
        raise ValueError(f"{column} not in data")  
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    len_initial = len(data)
    data_filtered = data.loc[(data[column] <= upper_bound) & (data[column] >= lower_bound)]
    len_final = len(data_filtered)
    outliers_removed = len_initial - len_final    
    print(f"# of outliers removed = {outliers_removed}")
    return data_filtered


class ClusterSimilarity(BaseEstimator,TransformerMixin):
  def __init__(self,n_clusters=10,gamma=1.,random_state=None,init='k-means++',n_init='auto'):
    self.random_state = random_state
    self.init = init
    self.n_init = n_init
    self.n_clusters = n_clusters
    self.gamma = gamma
    
  
  def fit(self,X,y=None,sample_weights=None):
    X = check_array(X)
    self.n_features_in_ = X.shape[1]
    self.kmeans_ = KMeans(n_clusters=self.n_clusters,random_state=self.random_state,init=self.init,n_init=self.n_init)
    assert self.n_features_in_ == X.shape[1]
  
    # Fit KMeans with sample weights if provided
    #sample weights are nothing but a list of integers from 1 to inf which indicates the weights of the individual data points . This allows the algorithm to perform a weighted average rather than a simple average.
    self.kmeans_.fit(X,**({} if sample_weights is None else {'sample_weight':sample_weights}))
    return self
  
  def transform(self,X):
    check_is_fitted(self)
    X = check_array(X)
    return rbf_kernel(X,Y=self.kmeans_.cluster_centers_,gamma=self.gamma)
  
  
  def get_feature_names_out(self,names=None):
    return [f"Cluster {i} similarity" for i in range(self.n_clusters)]




preprocessing = ColumnTransformer(
    transformers=[
        ("rooms_per_household", ratio_pipeline, ["total_rooms", "households"]),
        ("people_per_household", ratio_pipeline, ["population", "households"]),
        ("bedrooms_per_household", ratio_pipeline, ["total_bedrooms", "total_rooms"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
        ("cat_pipeline", catpipeline, ["ocean_proximity"])
    ],
    remainder=num_pipeline,force_int_remainder_cols=False
)



