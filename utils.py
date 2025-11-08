import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

# Lista de tus aves (orden debe coincidir con la salida del modelo)
BIRD_CLASSES = [
    "Accipiter_bicolor",
    "Ardea_cocoi",
    "Buteo_albigula",
    "Cathartes_burrovianus",
    "Chondrohierax_uncinatus",
    "Dryocopus_lineatus",
    "Egretta_thula",
    "Falco_columbarius",
    "Melanerpes_formicivorus",
    "Sarcoramphus_papa"
]

def predict(model, img, model_type="Xception"):
    """
    Devuelve predicciones de un modelo de clasificación de aves.
    
    Parameters:
    - model: tf.keras.Model ya cargado
    - img: PIL.Image
    - model_type: "Xception" o "VGG16"
    
    Returns:
    - Lista de tuplas: [(nombre_clase, probabilidad), ...]
    """
    try:
        # Redimensionar según modelo
        if model_type == "Xception":
            target_size = (299, 299)
            preprocess = xception_preprocess
        elif model_type == "VGG16":
            target_size = (224, 224)
            preprocess = vgg16_preprocess
        else:
            return []

        # Convertir PIL -> array numpy
        img_resized = img.resize(target_size)
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess(x)

        # Predicción
        preds = model.predict(x)[0]

        # Crear lista de tuplas (nombre, probabilidad)
        results = list(zip(BIRD_CLASSES, preds))
        return results

    except Exception as e:
        print("Error en predict():", e)
        return []
