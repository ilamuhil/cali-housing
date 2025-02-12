import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)

def save_model(model, path):
  joblib.dump(model, path)
  logging.info(f"Model saved to {path}")

def delete_model(path):
  if os.path.exists(path):
    os.remove(path)
    logging.info(f"Model deleted from {path}")
  else:
    logging.warning(f"No model found at {path}")

def load_model(path):
  if os.path.exists(path):
    model = joblib.load(path)
    logging.info(f"Model loaded from {path}")
    return model
  else:
    logging.warning(f"No model found at {path}")
    return None