import pandas as pd
import numpy
import json
from json import JSONEncoder
import io
import pickle
import lightgbm
import uvicorn
from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


# Create the app object
clf = pickle.load(open("model.pkl", "rb"))
# Create FastAPI instance
app = FastAPI()

origins = [
    "https://oc-p7-streamlit.herokuapp.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




#In this example, we try to serialize the NumPy Array into JSON String
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Create POST endpoint with path '/predict'
@app.post("/predict")
async def predict(file: bytes = File(...)):
    print('[+] Initiate Prediction')
    file_obj = io.BytesIO(file)
    sample_df = pd.read_csv(file_obj, index_col='SK_ID_CURR')

    # Generate predictions with best model (output is H2O frame)
    preds = clf.predict(sample_df)
    
    # Apply processing if dataset has ID column
    preds_final = {"Prediction": preds}

    # use dump() to write array into file
    encodedNumpyData = json.dumps(preds_final, cls=NumpyArrayEncoder) 

    return JSONResponse(content=encodedNumpyData)

@app.get("/")
async def main():
    content = """
    <body>
    <h2> Welcome to the End to End ML Model of customer Scoring</h2>
    <p> The customer Scoring model and FastAPI instances have been set up successfully </p>
    <p> You can view the FastAPI UI by heading to localhost:8000 </p>
    <p> Proceed to initialize the Streamlit UI (frontend/app.py) to submit prediction requests </p>
    </body>
    """

    return HTMLResponse(content=content)