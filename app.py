import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Dense
from PIL import Image

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
    return tf.keras.models.load_model(modelo_path, custom_objects={"SimulatedQuantumLayer": SimulatedQuantumLayer})

# =========================================
# Función para preparar la imagen
# =========================================
def preparar_imagen(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array.flatten()
    vector_128 = np.zeros(128)
    length = min(img_array.size, 128)
    vector_128[:length] = img_array[:length]
    return np.expand_dims(vector_128, axis=0)

# =========================================
# Interfaz principal
# =========================================
st.title("Clasificador de Aves - Modelo Cuántico")

uploaded_file = st.file_uploader("Sube una imagen del ave:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)
    st.write("Procesando imagen...")

    modelo = cargar_modelo("modelos/cuantico.keras")
    img_preparada = preparar_imagen(uploaded_file)
    predicciones = modelo.predict(img_preparada)

    st.write("Predicciones:")
    st.write(predicciones)
