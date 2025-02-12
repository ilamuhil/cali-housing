from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import logging
from model import load_model
from prometheus_fastapi_instrumentator import Instrumentator


MODEL_PATH = "models/best_model.pkl"

app = FastAPI(title="California Housing Price Prediction API",
    description="This API allows you to predict housing prices based on various features.",
    version="1.0.0",swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"})

# Define request and response models
class HousingModel(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

class Prediction(BaseModel):
    prediction: float
    

@app.post("/predict", response_model=Prediction)
def predict(request: HousingModel):
    try:
        # Load model from path 
        model = load_model(MODEL_PATH)
        if not model:
            logging.error("Model not found")
            raise HTTPException(status_code=500, detail="Model not found")
        
        # Make prediction
        test_data = pd.DataFrame([request.model_dump()])
        similarities = load_model("data/objects/cluster_simil.pkl")
        
        # Transform the geo features using similarities
        geo_features = similarities.transform(test_data[['longitude', 'latitude']].values)
        test_data.drop(['longitude', 'latitude'], axis=1, inplace=True)
        geo_df = pd.DataFrame(geo_features, columns=similarities.get_feature_names_out())
        test_data = pd.concat([test_data, geo_df], axis=1)  
        prediction = model.best_estimator_.predict(test_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        print(e)
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {
        "Hello": "Welcome to the California housing price prediction API!",
        "information": "Please make a POST request to /predict with the required fields to get the prediction.",
        "fields": [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
            "median_house_value", "ocean_proximity"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    Instrumentator().instrument(app).expose(app)
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
   