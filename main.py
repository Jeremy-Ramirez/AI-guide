import os
import uuid
import numpy as np
import shutil
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("LOCAL_ORIGIN_FRONTEND"),os.getenv("LOCAL_HOST_ORIGIN_FRONTEND")], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# FastAPI will serve audios in  /audios/<filename>
app.mount("/audios", StaticFiles(directory="media/audios"), name="audios")


# Ensure the media directory exists
MEDIA_DIR = Path("media/uploads")
MEDIA_DIR.mkdir(exist_ok=True)


def save_image_to_media(file: UploadFile) -> str:
    """
    Save an image to the media directory with a unique name.

    Args:
        file: The uploaded image file

    Returns:
        str: The path where the image was saved
    """
    # Generate a unique name to avoid conflicts
    file_extension = Path(file.filename).suffix if file.filename else ".jpg"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = MEDIA_DIR / unique_filename

    # Save the file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Reset the file pointer for future use if necessary
        file.file.seek(0)

        return str(file_path)
    except Exception as e:
        raise Exception(f"Error saving the image: {str(e)}")


def get_art_category(index):

    name = ""
    audio = ""

    match index:
        case 0:
            name = "Cacao"
            audio = "cacao.mp3"
        case 1:
            name = "Metate"
            audio = "metate.mp3"
        case 2:
            name = "Molinillo"
            audio = "molinillo.mp3"
        case 3:
            name = "Mortero"
            audio = "mortero.mp3"
        case 4:
            name = "Silla con forma de U"
            audio = "silla.mp3"

    return name, audio


def predict_image(file: UploadFile):
    """
    Function to process and predict an image.
    First save the image in the media directory.
    """
    try:
        # Save the image in the media directory
        saved_path = save_image_to_media(file)

        altura = 100
        anchura = 100
        modelo = "modelo/modelo.h5"
        pesos = "modelo/pesos.h5"
        cnn = load_model(modelo)
        cnn.load_weights(pesos)

        x = load_img(saved_path, target_size=(anchura, altura))
        print(x)  # Image, mode=rgb, size=100x100
        x = img_to_array(x)
        # convert a PIL image instance to a numpy array that contains pixels
        print(x)
        x = np.expand_dims(x, axis=0)
        # in our axis 0 (first dimension), we want to add an extra dimension, to process our information
        print(x)
        # call our network and want to predict, returns an array of 2 dimensions
        arreglo = cnn.predict(x)
        print(arreglo)  # [1,0,0,0,0]
        # we only need 1 dimension, the one that brings the information
        resultado = arreglo[0]
        # presents the array with values of 0 and 1 as one hot encoding
        print(resultado)
        # returns the index of the maximum value of the array
        respuesta = np.argmax(resultado)
        print(respuesta)

        name, audio = get_art_category(respuesta)

        audio_url = f"/audios/{audio}"

        data = {"nombre": name, "audio": audio_url}

        return {"data": data}
    except Exception as e:
        return {"error": str(e), "message": "Error processing the image"}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    response = predict_image(file)
    return response
