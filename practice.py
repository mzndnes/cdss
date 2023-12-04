import pandas as pd
import requests
from fastapi import FastAPI
from models import *
import uvicorn
from retinopathy.retinopathy import RetinoPathy
from sklearn.model_selection import train_test_split
from xray_analysis.xray_analysis import XrayAnalysis
from heart_disease.heart_disease import HeartDisease
from clinical_notes.clinical_notes import ClinicalData
import numpy as np
from xray_analysis.util import compute_gradcam,load_image
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf

app=FastAPI()

@app.get('/')
def index():
        return {"message": "Welcome to cdss"}