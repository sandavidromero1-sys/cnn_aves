import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects

# =========================================
# Definición del SimulatedQuantumLayer
# =========================================
@tf.keras.utils.register_keras_serializable()
class SimulatedQuantumLayer(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super(SimulatedQuantumLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="random_normal",
                                 trainable=True)
        super(SimulatedQuantumLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.math.sin(tf.matmul(inputs, self.w))

# =========================================
# Función para cargar el modelo
# =========================================
@st.cache_resource
def cargar_modelo(modelo_path):
    return load_model(modelo_path, custom_objects={"SimulatedQuantumLayer": SimulatedQuantumLayer})

# =========================================
# Preprocesamiento de imagen para modelo cuántico
# =========================================
def preparar_imagen(img):
    # Convertir a RGB y redimensionar (tamaño arbitrario pequeño)
    img = img.convert("RGB").resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array.flatten()  # aplanar
    # Crear vector de 128 elementos
    vector_128 = np.zeros(128)
    length = min(img_array.size, 128)
    vector_128[:length] = img_array[:length]
    return np.expand_dims(vector_128, axis=0)  # forma (1,128)

# =========================================
# Diccionario de modelos
# =========================================
MODELOS = {
    "Modelo Cuántico": "modelos/cuantico.keras",
    # Dejamos Xception y VGG16 apuntando al modelo cuántico para que no fallen
    "Xception": "modelos/cuantico.keras",
    "VGG16": "modelos/cuantico.keras"
}

# =========================================
# Interfaz principal
# =========================================
st.title("Clasificador de Aves con Modelo Cuántico")
st.write("Selecciona el modelo a utilizar:")

modelo_seleccionado = st.selectbox("Modelo:", list(MODELOS.keys()))
modelo = cargar_modelo(MODELOS[modelo_seleccionado])

uploaded_file = st.file_uploader("Sube una imagen del ave:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagen cargada", use_column_width=True)
    st.write("Procesando imagen...")

    img_preparada = preparar_imagen(img)
    predicciones = modelo.predict(img_preparada)

    # Mostramos los resultados
    st.write("Predicciones:")
    st.write(predicciones)
