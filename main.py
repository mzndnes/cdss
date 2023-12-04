import os.path
from io import BytesIO
import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, Request, File
from starlette.responses import JSONResponse
from models import *
import uvicorn
from retinopathy.retinopathy import RetinoPathy
from sklearn.model_selection import train_test_split
from xray_analysis.xray_analysis import XrayAnalysis
from coronary_artery_disease.coronary_artery_disease import CoronaryArtery
from clinical_notes.clinical_notes import ClinicalData
from pneumonia.pneumonia_analysis import PneumoniaAnalysis
from chest_xray_recognition.chest_xray_recognition import ChestXrayRecognition
from diabetes.diabetes_prediction import Classify as diabetes_model
from cardio_vascular.cardio_vascular import CardioVascular
import numpy as np
from xray_analysis.util import compute_gradcam,load_image
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import tensorflow as tf
import wget
from eye_cataract_model.Cataract_prediction import Classify as eye_model
from brain_mri_model.MRI import Classify as mri_model
from mri_identification.mri_identidication import identify as identify_mri
app=FastAPI()



origins = ["*"]
# origins = ['http://localhost:8000','http://192.168.101.17:8000']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def index():
        return {"message": "Welcome to cdss"}

@app.get('/send_to_predict_retinopathy')
def send():
    url = 'http://127.0.0.1:8000/predict_retinopathy'
    payload = {'Age': 77.19633951,
               'Systolic_BP': 85.28,
               'Diastolic_BP':80.02,
               'Cholesterol':79.95,
               'hospital_id':1,
               'patient_id':1,
               'encounter_number':1}

    response = requests.post(url, json=payload)
    result=response.json()
    return result

@app.post('/predict_retinopathy',response_model=RetinopathyResult)
def predict(res:RetinopathyData)->any:
    data = res.dict()
    hospital_id=data.pop('hospital_id')
    patient_id=data.pop('patient_id')
    encounter_number=data.pop('encounter_number')
    is_diabetic=data.pop('is_diabetic')

    df = pd.DataFrame(data, index=[0])
    classifier = RetinoPathy()
    X = pd.read_csv('retinopathy/X_data.csv', index_col=0)
    y_df = pd.read_csv('retinopathy/y_data.csv', index_col=0)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y_df, train_size=0.75, random_state=0)
    X_train, X_test = classifier.standardize_train_test(X_train_raw, X_test_raw)
    df = classifier.standardize_data(df)
    label = classifier.classify(df)
    # result=get_result(patient_id,hospital_id,encounter_number)
    # r_risk=round(label[0][1],3)

    prob=round(label[0][1], 6)
    data["prob"] = prob
    if prob<0.5:
        msg="Low risk of Daibetic Retinopathy "
    elif prob<0.8:
        msg="Moderate risk of Diabetic Retinopathy "
    else:
        msg="High risk of Diabetic Retinopathy "
    data['is_diabetic']=is_diabetic
    data['hospital_id']=hospital_id
    data['patient_id']=patient_id
    data['encounter_number']=encounter_number
    data["msg"]=msg
    # data["msg"]=["No risk of Daibetic Retinopathy","Low risk of Diabetic Retinopathy","Medium risk of Diabetic Retinopathy", "High risk of Diabetic Retinopathy","Patient has Diabetic Retinopathy"]
    # data["range"]=["[0,0.02]", "(0.02,0.4]","(0.4,0.6]","(0.6,0.98]","(0.98,1]"]

    return data

@app.post('/chest_xray_recognition', response_model=ChestXrayResult)
def predict(res:ChestXrayData)-> any:
    data=res.dict()
    img=data['src_image']
    classifier = ChestXrayRecognition()
    response = None
    try:
        response = requests.get(img, verify=False)
    except:
        print("Site not rechable")
    img_bytes = BytesIO(response.content)
    processed_input = classifier.prepare_input(img_bytes)
    prob = classifier.classify(processed_input)

    label=np.argmax(prob,axis=1)
    data['prob']=label
    if label==0:
        msg="Image is chest X-ray"
    else:
        msg="Image is not chest X-ray"
    data['msg']=msg
    return data

@app.post('/predict_pneumonia', response_model=PneumoniaResult)
def predict(res:PneumoniaData)-> any:
    data=res.dict()
    img=data['src_image']
    classifier = PneumoniaAnalysis()
    response = None
    try:
        response = requests.get(img, verify=False)
    except:
        print("Site not rechable")
    img_bytes = BytesIO(response.content)
    processed_input = classifier.prepare_input(img_bytes)

    label = classifier.classify(processed_input)
    prob = round(label[0][0],6)
    data['prob']=prob
    if prob<0.5:
        msg="Low risk of Pneumonia."
    elif prob<0.8:
        msg="Moderate risk of Pneumonia."
    else:
        msg="High risk of Pneumonia"
    data['msg']=msg
    # data['msg']=['No risk of Pneumonia','Patient is Pneumonia Positive', 'AI is not confident']
    # data['range']=['[0,0.44]','[0.99,1]','(0.44,0.99)']

    return data

@app.get('/send_to_analyse_xray')
def send():

    url = 'http://127.0.0.1:8000/analyse_xray'
    data = {'Image': 'xray_analysis/00005410_000.png',
            'patient_id':1,
            'hospital_id':1,
            'encounter_number':1}


    response = requests.post(url, json=data)
    result=response.json()
    return result

@app.post('/analyse_xray', response_model=XrayAnalysiss)
def predict(res:XrayData)-> any:
    data=res.dict()
    patient_id=data['patient_id']
    hospital_id=data['hospital_id']
    encounter_number=data['encounter_number']
    img=data['src_image']


    # df=pd.DataFrame(data,index=[0])
    classifier =XrayAnalysis()
    # response=FileResponse(img)

    # print(uploaded_file.filename)
    # print(uploaded_file.path)
    # print(uploaded_file.headers)
    # with open(uploaded_file, "wb+") as file_object:
    #     shutil.copyfileobj(uploaded_file.filename, file_object)


    labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia',
              'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia',
              'Fibrosis', 'Edema', 'Consolidation']

    # img_name=wget.download(img)
    response=None
    try:
        response = requests.get(img,verify=False)
    except:
        print("Site not rechable")
    img_bytes = BytesIO(response.content)
    # if response.headers['content-type'] == 'image/png':
    #     img_name += '.png'
    # elif response.headers['content-type'] == 'image/jpeg':
    #     img_name += '.jpg'
    # with open(img_name, "wb") as f:
    #     f.write(response.content)
    # IMAGE_DIR = "xray_analysis/train_images/"
    # classifier.get_train_generator(train_df, IMAGE_DIR, "Image", labels)
    # test_generator = classifier.get_test_generator(df, IMAGE_DIR, "Image")

    # x = test_generator.__getitem__(0)
    # img = x[0].reshape(1, 320, 320, 3)

    # preds,imgs=compute_gradcam(classifier.model,img_name,patient_id, hospital_id,encounter_number,labels)
    preds,imgs= compute_gradcam(classifier.model, img_bytes, patient_id, hospital_id,
                                  encounter_number, labels)

    # result=get_result(patient_id,hospital_id,encounter_number)

    # result['preds']=list(preds)
    data['heatmap_images']=list(imgs)
    # data['heatmap_images'] = []

    for key,value in zip(labels,preds):
        data[key]=value

    return data


def get_result(patient_id,hospital_id,encounter_number):
    result = dict()
    result['patient_id'] = patient_id
    result['hospital_id'] = hospital_id
    result['encounter_number'] = encounter_number
    result['is_diabetic']=1
    return result


@app.post('/cardio_vascular',response_model=CardioResult)
def predict(res:CardioData)->any:
    data=res.dict()
    patient_id = data.pop('patient_id')
    hospital_id = data.pop('hospital_id')
    encounter_number = data.pop('encounter_number')

    classifier = CardioVascular()
    processed_input = classifier.prepare_input(data)
    label = classifier.classify(processed_input)
    data['patient_id']=patient_id
    data['hospital_id']=hospital_id
    data['encounter_number']=encounter_number

    prob=round(label[0][0],6)
    data['prob']=prob
    if prob<0.5:
        msg="Low risk of Heart Disease."
    elif prob<0.8:
        msg="Moderate risk of Heart Disease."
    else:
        msg="High risk of Heart Disease."
    data['msg']=msg
    # data['range']=['[0,0.3)','[0.3,0.6)','[0.6,1]']
    # data['msg']=['Low risk of Heart Disease','Medium risk of Heart Disease','High risk of Heart Disease']

    return data

@app.get('/send_to_analyse_notes')
def send():
    url = 'http://127.0.0.1:8000/analyse_notes'
    payload = {'notes': "Patient presented with symptoms of fever, dry cough and shortness of breathe",
               'hospital_id':1,
               'patient_id':1,
               'encounter_number':1}

    response = requests.post(url, json=payload)
    result=response.json()
    return result

@app.post('/analyse_notes',response_model=ClinicalNotesResult)
def predict(res:ClinicalNotes)->any:
    data=res.dict()
    # patient_id = data['patient_id']
    # hospital_id = data['hospital_id']
    # encounter_number = data['encounter_number']
    notes=data['notes']
    classifier=ClinicalData()

    doc=classifier.classify(notes)

    # result=get_result(patient_id,hospital_id,encounter_number)

    for ent in doc.entities:
        entity=ent.type.lower()
        if entity in data:
            data[entity]+="," + ent.text
        else:
            data[entity] = ent.text

    entities=['treatment','problem','test']
    # entities=set(entities)
    for entity in entities:
        if entity not in data:
            data[entity]=""
    return data

@app.post('/predict_mri', response_model=mri_results)
def predict(res: mri_data, request: Request) -> any:
    data = res.dict()
    pid = data['patient_id']
    hid = data['hospital_id']
    encounter_id = data['encounter_number']
    classify_mri = mri_model()
    file_name = classify_mri.predict(data['files'], encounter_id)
    # result = get_result(pid, hid, encounter_id)
    ip_address=request.base_url.hostname
    # file_url = f'http://127.0.0.1:8000/download?file_path={file_name}'
    file_url = f'http://{ip_address}:81/download?file_path={file_name}'
    data['file_name'] = file_url
    return data

#
# @app.get('/send_to_eye')
# def send():
#     url = 'http://127.0.0.1:8000/predict_eye'
#     payload = Request.json()
#     s = requests.Session()
#     response = s.post(url, json=payload)
#     if response.status_code == 200:
#         result = response.json()
#         return result


@app.post('/predict_eye_cataract', response_model=eye_results)
def predict(res: eye_data, request: Request):
    data = res.dict()
    encounter_number = data['encounter_number']
    eye_type =data['eye_type']
    classify_eye = eye_model()
    prediction_float, file_name = classify_eye.predict(data['file_name'],eye_type, encounter_number, 'predict')
    data['normal']=prediction_float[0,0]
    data['cataract']=prediction_float[0,1]
    # file_response = FileResponse(file_name)
    ip_address=request.base_url.hostname
    file_url = f'http://{ip_address}:81/download?file_path={file_name}'
    data['output_file']= file_url
    return data


@app.get('/download')
def download_file(file_path: str):
    local_file_path = file_path
    if os.path.exists(local_file_path):
        filename = os.path.basename(local_file_path)
        return FileResponse(local_file_path, filename=filename)
    else:
        return JSONResponse(content={"error": "File not found"}, status_code=404)





# @app.get('/send_to_test')
# def send():
#     url = 'https://127.0.0.1:8000/test_images'
#     payload = Request.json()
#     s = requests.Session()
#     response = s.post(url, json=payload)
#     if response.status_code == 200:
#         result = response.json()
#         return result

@app.post('/test_images', response_model=image_test_result)
def predict(res: image_test_data) -> any:
    data = res.dict()
    image_category = data['image_category']
    if image_category == 'eye':
        classify_eye = eye_model()
        test_result = classify_eye.predict(data['file_name'],0, 11111111, 'test')
        data['is_eye'] = test_result
        data['image_category'] = image_category
        return data

@app.post('/predict_diabetes', response_model=diabetes_results)
def predict(res: diabetes_data) -> any:
    data = res.dict()
    temp=dict(data)
    if data['gender']==2:
        data['gender']=1
    height=data['height']
    weight=data['weight']
    BMI=weight/((height*0.0254)*(height*0.0254))
    data.pop('height')
    data.pop('weight')
    data['bmi']=BMI
    data.pop('hospital_id')
    data.pop('patient_id')
    data.pop('encounter_number')
    # print(data)
    predict_diabetes = diabetes_model()
    prediction = predict_diabetes.predict(data)
    temp['normal'] = prediction[0,0]
    temp['pre_diabetic']=prediction[0,1]
    temp['diabetic']=prediction[0,2]
    return temp


@app.post('/mri_identification', response_model=mri_identification_result)
def predict(res: mri_identification_data) -> any:
    data = res.dict()
    files=data['files']
    identify=identify_mri()
    prediction=identify.predict(files)
    data['is_mri']=prediction
    return data


if __name__=='__main__':
    uvicorn.run(app,host="0.0.0.0",port=8000, timeout_keep_alive=60)


