import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import gdown

# =====================================================
# Clase personalizada para el modelo cuántico
# =====================================================
from keras.saving import register_keras_serializable

@register_keras_serializable()
class SimulatedQuantumLayer(Layer):
    def __init__(self, units=64, **kwargs):
        super(SimulatedQuantumLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="quantum_weight",
            shape=(int(input_shape[-1]), self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="quantum_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        super(SimulatedQuantumLayer, self).build(input_shape)

    def call(self, inputs):
        salida = tf.matmul(inputs, self.w) + self.bias
        ruido = tf.random.normal(shape=tf.shape(salida), mean=0.0, stddev=0.01)
        return tf.nn.tanh(salida + ruido)

    def get_config(self):
        config = super(SimulatedQuantumLayer, self).get_config()
        config.update({"units": self.units})
        return config

# =====================================================
# Función para descargar modelos desde Google Drive
# =====================================================
def descargar_modelo(nombre_modelo):
    rutas_drive = {
        "cuantico": "https://drive.google.com/uc?id=1XxRzz3sp_SDKFzxds7A3gFsVaNpkMcfr",
        "vgg16": "https://drive.google.com/uc?id=1XxRzz3sp_SDKFzxds7A3gFsVaNpkMcfr",  # Cambia por tu VGG16 real
        "xception": "https://drive.google.com/uc?id=1XxRzz3sp_SDKFzxds7A3gFsVaNpkMcfr"  # Cambia por tu Xception real
    }
    carpeta_modelos = "modelos"
    if not os.path.exists(carpeta_modelos):
        os.makedirs(carpeta_modelos)
    ruta_local = os.path.join(carpeta_modelos, f"{nombre_modelo}.keras")
    if not os.path.exists(ruta_local):
        gdown.download(rutas_drive[nombre_modelo], ruta_local, quiet=False)
    return ruta_local

# =====================================================
# Función para cargar modelo
# =====================================================
@st.cache_resource
def cargar_modelo(nombre_modelo):
    modelo_path = descargar_modelo(nombre_modelo)
    if "cuantico" in nombre_modelo.lower():
        return tf.keras.models.load_model(modelo_path, custom_objects={"SimulatedQuantumLayer": SimulatedQuantumLayer})
    else:
        return tf.keras.models.load_model(modelo_path)

# =====================================================
# Función para preparar imagen
# =====================================================
def preparar_imagen(img, modelo_nombre):
    if "xception" in modelo_nombre.lower():
        target_size = (299, 299)
    else:
        target_size = (224, 224)
    img = image.img_to_array(img.resize(target_size))
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# =====================================================
# Diccionario de modelos
# =====================================================
MODELOS = {
    "Cuántico": "cuantico",
    "VGG16": "vgg16",
    "Xception": "xception"
}

# =====================================================
# INTERFAZ PRINCIPAL
# =====================================================
st.title("Clasificador de Aves")
st.write("Selecciona el modelo a utilizar y sube una imagen de ave.")

modelo_seleccionado = st.selectbox("Selecciona el modelo a utilizar:", list(MODELOS.keys()))
modelo = cargar_modelo(MODELOS[modelo_seleccionado])

uploaded_file = st.file_uploader("Sube una imagen del ave:", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    from PIL import Image
    img = Image.open(uploaded_file).convert("RGB")
    img_array = preparar_imagen(img, MODELOS[modelo_seleccionado])

    pred = modelo.predict(img_array)[0]
    clases = [f"Ave {i+1}" for i in range(len(pred))]

    top3_idx = np.argsort(pred)[-3:][::-1]
    st.write("Predicciones:")
    for i in top3_idx:
        st.write(f"{clases[i]}: {pred[i]*100:.2f}%")

    # Gráfico de barras
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bar([clases[i] for i in top3_idx], [pred[i]*100 for i in top3_idx], color="skyblue")
    ax.set_ylabel("Probabilidad (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)
