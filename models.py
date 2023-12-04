from pydantic import BaseModel,model_validator, constr, conint, confloat
from typing import Union
import requests
from validations import *
from typing import List


class Base_Data(BaseModel):
    patient_id: str
    hospital_id: str
    encounter_number: constr(min_length=8, max_length=15)= 'string'


class RetinopathyData(Base_Data):
    is_diabetic:conint(ge=0, le=1) = 'int'
    Age:Union[confloat(ge=0.0, le=200.0),conint(ge=0, le=200)] = 'int/float'
    Systolic_BP:Union[confloat(ge=0.0, le=300.0),conint(ge=0, le=300)] = 'int/float'
    Diastolic_BP:Union[confloat(ge=0.0, le=200.0),conint(ge=0, le=200)] = 'int/float'
    Cholesterol:Union[confloat(ge=0.0, le=300.0),conint(ge=0, le=300)] = 'int/float'

    @model_validator(mode='after')
    def validata_data(self) -> 'RetinopathyData':
        validate_RetinopathyData(self)
        return self

class RetinopathyResult(RetinopathyData):
    prob: float
    msg:str

class ChestXrayData(Base_Data):
    src_image: constr(min_length=1) = 'string'

    @model_validator(mode='after')
    def validata_data(self) -> 'ChestXrayData':
        validate_chest_xray_recognition(self)
        return self

class ChestXrayResult(ChestXrayData):
    prob:int
    msg:str


class PneumoniaData(Base_Data):
    is_chest_xray:conint(ge=0, le=1) = 'int'
    src_image: constr(min_length=1) = 'string'

    @model_validator(mode='after')
    def validate_data(self) -> 'PneumoniaData':
        validate_Pneumionia(self)
        return self


class PneumoniaResult(PneumoniaData):
    prob: float
    msg:str



class XrayData(Base_Data):
    src_image:constr(min_length=1) = 'string'


class XrayAnalysiss(Base_Data):
    # preds:list[float]=[]
    heatmap_images:list[str]=[]
    Cardiomegaly:float
    Emphysema:float
    Effusion:float
    Hernia:float
    Infiltration:float
    Mass:float
    Nodule:float
    Atelectasis:float
    Pneumothorax:float
    Pleural_Thickening:float
    Pneumonia:float
    Fibrosis:float
    Edema:float
    Consolidation:float
    src_image:str
    
class CardioData(Base_Data):
    age: conint(ge=1, le=200)= 'int'
    gender: conint(ge=0, le=1)= 'int'
    height: conint(ge=1, le=300)= '(1 to 300)'
    # rest_blood_pressure: conint(ge=80, le=200)= 'int'
    weight:conint(ge=1, le=500)= 'int'
    ap_hi: conint(ge=1, le=500) = 'int'
    ap_lo: conint(ge=1, le=300) = 'int'
    cholesterol: conint(ge=1, le=3) = 'int'
    gluc: conint(ge=1, le=3) = '(1- 3)'
    smoke: conint(ge=0, le=1) = '(0,1)'
    alco: conint(ge=0, le=1)='(0,1)'
    active: conint(ge=0, le=1)='(0 , 1)'


    # @model_validator(mode='after')
    # def validate_data(self) -> 'CoronaryData':
    #     validate_CardioData(self)
    #     return self

class CardioResult(CardioData):
    prob:float
    msg:str
class ClinicalNotes(Base_Data):
    notes:constr(min_length=1, max_length=10000) ='string'


class ClinicalNotesResult(ClinicalNotes):
    problem:str
    test:str
    treatment:str

class mri_data(Base_Data):
    files: List[constr(min_length=1)]


    @model_validator(mode='after')
    def validate_data(self) -> 'mri_data':
        if len(self.files)!=3:
            raise ValueError("File number mismatched")
        validate_mri(self)
        return self



class mri_results(mri_data):
    file_name: str


class eye_data(Base_Data):
    file_name: constr(min_length=1) = 'string'
    eye_type: conint(ge=0, le=1) = "0 for left, 1 for right"

    @model_validator(mode='after')
    def validate_data(self) -> 'eye_data':
        validate_eyedata(self)
        return self


class eye_results(eye_data):
    cataract:float
    normal:float
    output_file: str

class image_test_data(BaseModel):
    file_name: constr(min_length=1) = 'string'
    image_category: constr(min_length=1, max_length=20) ='string'

    @model_validator(mode='after')
    def validate_data(self) -> 'image_test_data':
        validate_image_check(self)
        return self


class image_test_result(image_test_data):
    is_eye: int


class diabetes_data(Base_Data):
    gender: conint(ge=0, le=2) = 'int'
    age:  conint(ge=0, le=120) ='int'
    hba1c: confloat(ge=1.0, le=100.0)= 'float'
    weight: confloat(ge=0.1, le=500.0)= 'in_kg'
    height:confloat(ge=0.1, le=300)= "in_inch"


class diabetes_results(diabetes_data):
    normal: float
    pre_diabetic:float
    diabetic:float


class mri_identification_data(Base_Data):
    files: List[str]

    @model_validator(mode='after')
    def validate_data(self) -> 'mri_identification_data':
        validate_mri(self)
        return self

class mri_identification_result(mri_identification_data):
    is_mri: list