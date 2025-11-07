import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Layer
import os
import gdown
from PIL import Image

# =====================================================
# Clase personalizada para el modelo cuántico
# =====================================================
class SimulatedQuantumLayer(Layer):
    def __init__(self, units=64, **kwargs):
        super(SimulatedQuantumLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(int(input_shape[-1]), self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super(SimulatedQuantumLayer, self).build(input_shape)

    def call(self, inputs):
        salida = tf.matmul(inputs, self.w) + self.bias
        ruido = tf.random.normal(shape=tf.shape(salida), mean=0.0, stddev=0.01)
        return tf.nn.tanh(salida + ruido)

# =====================================================
# Función para descargar modelo
# =====================================================
def descargar_modelo(nombre_modelo):
    # Usamos la misma ruta del modelo cuántico para todos
    ruta_drive_cuantico = "https://drive.google.com/uc?id=1XxRzz3sp_SDKFzxds7A3gFsVaNpkMcfr"
    carpeta_modelos = "modelos"
    if not os.path.exists(carpeta_modelos):
        os.makedirs(carpeta_modelos)
    ruta_local = os.path.join(carpeta_modelos, "cuantico.keras")
    if not os.path.exists(ruta_local):
        gdown.download(ruta_drive_cuantico, ruta_local, quiet=False)
    return ruta_local

# =====================================================
# Función para cargar modelo
# =====================================================
@st.cache_resource
def cargar_modelo(nombre_modelo):
    modelo_path = descargar_modelo(nombre_modelo)
    return tf.keras.models.load_model(modelo_path, custom_objects={"SimulatedQuantumLayer": SimulatedQuantumLayer})

# =====================================================
# Función para preparar imagen
# =====================================================
def preparar_imagen(img):
    target_size = (224, 224)
    img = image.img_to_array(img.resize(target_size))
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# =====================================================
# Diccionario de modelos (todos apuntan al cuántico)
# =====================================================
MODELOS = {
    "Cuántico": "cuantico",
    "VGG16": "cuantico",
    "Xception": "cuantico"
}

# =====================================================
# INTERFAZ PRINCIPAL
# =====================================================
st.title("Clasificador de Aves (Todos usando el Cuántico)")
st.write("Sube una imagen de ave y el modelo cuántico hará la predicción.")

modelo_seleccionado = st.selectbox("Selecciona el modelo a utilizar:", list(MODELOS.keys()))
modelo = cargar_modelo(MODELOS[modelo_seleccionado])

uploaded_file = st.file_uploader("Sube una imagen del ave:", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = preparar_imagen(img)

    pred = modelo.predict(img_array)[0]
    clases = [f"Ave {i+1}" for i in range(len(pred))]

    top3_idx = np.argsort(pred)[-3:][::-1]
    st.write("Predicciones:")
    for i in top3_idx:
        st.write(f"{clases[i]}: {pred[i]*100:.2f}%")
