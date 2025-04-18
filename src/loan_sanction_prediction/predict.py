from pathlib import Path
import joblib
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import pandas as pd

from loan_sanction_prediction.utils import setup_logger
app = FastAPI()

model_dir = Path("models")
pl = joblib.load(model_dir / "pl.joblib")
le = joblib.load(model_dir / "le.joblib")

class InputData(BaseModel):
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    dependents: str
    property_area: str
    gender: str
    married: str
    education: str
    self_employed: str
    loan_amount_term: int
    credit_history: int

@app.post("/prediction")
def predict(data: InputData):
    log = setup_logger(Path(__file__).stem)
    data_dict = data.model_dump()

    # Process user inputs
    try:
        data_dict["married"] = "yes" if data_dict["married"] == "Married" else "no"
        if data_dict["property_area"] == "Semi-urban":
            data_dict["property_area"] = "semiurban"
        for feature in ["gender", "education", "self_employed", "property_area"]:
            data_dict[feature] = data_dict[feature].lower().replace(" ", "_")
        log.success("Successfully processed user inputs")
    except Exception:
        log.exception("Failed to process user inputs")
        raise HTTPException(status_code=400, detail="Input processing failed")

    try:
        input_df = pd.DataFrame([data_dict])
        prediction = pl.predict(input_df)
        result = le.inverse_transform(prediction)[0]
        log.success("Prediction successful")
        return {"result": 1 if result == 'yes' else 0}
    except Exception as e:
        log.exception(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
