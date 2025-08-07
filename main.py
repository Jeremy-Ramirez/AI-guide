import os
import uuid
import numpy as np
import shutil
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.98:3000"],  # tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Con esto, FastAPI servirá cualquier archivo en /audios/<filename>. Por ejemplo:
app.mount("/audios", StaticFiles(directory="media/audios"), name="audios")


# Asegurar que el directorio media existe
MEDIA_DIR = Path("media/uploads")
MEDIA_DIR.mkdir(exist_ok=True)


def save_image_to_media(file: UploadFile) -> str:
    """
    Guarda una imagen en el directorio media con un nombre único.

    Args:
        file: El archivo de imagen subido

    Returns:
        str: La ruta donde se guardó la imagen
    """
    # Generar un nombre único para evitar conflictos
    file_extension = Path(file.filename).suffix if file.filename else ".jpg"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = MEDIA_DIR / unique_filename

    # Guardar el archivo
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Resetear el puntero del archivo para uso posterior si es necesario
        file.file.seek(0)

        return str(file_path)
    except Exception as e:
        raise Exception(f"Error al guardar la imagen: {str(e)}")


def get_art_category(index):

    name = ""
    audio = ""

    match index:
        case 0:
            name = "cacao"
            audio = "cacao.mp3"
        case 1:
            name = "metate"
            audio = "metate.mp3"
        case 2:
            name = "molinillo"
            audio = "molinillo.mp3"
        case 3:
            name = "mortero"
            audio = "mortero.mp3"
        case 4:
            name = "silla con forma de U"
            audio = "silla.mp3"

    return name, audio


def predict_image(file: UploadFile):
    """
    Función para procesar y predecir una imagen.
    Primero guarda la imagen en el directorio media.
    """
    try:
        # Guardar la imagen en el directorio media
        saved_path = save_image_to_media(file)

        altura = 100
        anchura = 100
        modelo = "modelo/modelo.h5"
        pesos = "modelo/pesos.h5"
        cnn = load_model(modelo)
        cnn.load_weights(pesos)

        x = load_img(saved_path, target_size=(anchura, altura))
        print(x)  # Imagen, modo=rgb, size=100x100
        x = img_to_array(x)
        # convierte una instancia de imagen PIL a un arreglo de numpy que contiene pixeles
        print(x)
        x = np.expand_dims(x, axis=0)
        # en nuestro eje 0 (primera dimension), queremos añadir una dimension extra, para procesar nuestra información
        print(x)
        # llamamos a nuestra red y queremos predecir, regresa un arreglo de 2 dimensiones
        arreglo = cnn.predict(x)
        print(arreglo)  # [1,0,0,0,0]
        # solo necesitamos 1 dimension, la que trae la información
        resultado = arreglo[0]
        # nos presenta el arreglo con valores de 0 y 1 como one hot encoding
        print(resultado)
        # retorna el índice del máximo valor del arreglo
        respuesta = np.argmax(resultado)
        print(respuesta)

        name, audio = get_art_category(respuesta)

        audio_url = f"/audios/{audio}"

        data = {"nombre": name, "audio": audio_url}

        return {"data": data}
    except Exception as e:
        return {"error": str(e), "message": "Error al procesar la imagen"}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    response = predict_image(file)
    return response
