import requests
from urllib.parse import urlparse
import os


supported_image_extensions = (".png", ".jpeg", ".jpg")
supported_mri_extensions = (".nii",".nii.gz")
#
# def validate_length(min_length,max_length,holder,data):
#     if not data or len(data) < min_length or len(data) > max_length:
#         raise ValueError(f"{holder} must be of length between {min_length} and {max_length}")
#
# def validate_value(min_value,max_value,holder,data):
#     if data <= min_value or data >= max_value:
#         raise ValueError(f"{holder} value must be in between {min_value} and {max_value} ")

# def validate_file_url(src_image: str):
#     try:
#         if not requests.get(src_image):
#             raise ValueError("Could not read the image")
#     except requests.exceptions.RequestException as e:
#         raise ValueError("Could not access the URL")

def validate_extension(holder,location,extensions):
    path = urlparse(location).path
    file_extension = os.path.splitext(path)[1]
    if file_extension.lower() not in extensions:
        raise ValueError(f"{holder} extension invalid: supported extension are {extensions}")

def validate_list_extension(holder,location,extensions):
    path = urlparse(location).path
    file_extension = os.path.splitext(path)[1]
    for extension in extensions:
        if file_extension.lower() not in extension:
            raise ValueError(f"{holder} extension invalid: supported extension are {extensions}")


def validate_condition(holder,data, condition):
    if data != condition:
        raise ValueError(f"{holder} must be {condition} to apply AI ")


def validate_blood_pressure(diastolic_bp, systolic_bp):
    if diastolic_bp > systolic_bp:
         raise ValueError("Diastolic bp cannot be greater than Systolic bp")


def validate_RetinopathyData(pointer):
    validate_condition('is_diabetic',pointer.is_diabetic,1)
    validate_blood_pressure(pointer.Diastolic_BP, pointer.Systolic_BP)

def validate_Pneumionia(pointer):
    validate_extension('src_image', pointer.src_image, supported_image_extensions)
    validate_condition('is_chest_xray', pointer.is_chest_xray, 1)

def validate_mri(pointer):
    for index, file in enumerate(pointer.files):
        validate_list_extension(f"file{index}", file, supported_mri_extensions)

def validate_eyedata(pointer):
    validate_extension('file_name', pointer.file_name, supported_image_extensions)

def validate_image_check(pointer):
    validate_extension('file_name', pointer.file_name, supported_image_extensions)

def validate_chest_xray_recognition(pointer):
    validate_extension('src_image', pointer.src_image, supported_image_extensions)

# def validate_CoronaryData(pointer):
#     accepted_values=(3,6,7)
#     if pointer.thal not in accepted_values:
#         raise ValueError('thal value must be 3, 6, or 7')
