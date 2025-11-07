import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Layer
from PIL import Image
import os
import plotly.express as px
import gdown

# =====================================================
# CAPA PERSONALIZADA PARA EL MODELO CUÁNTICO
# =====================================================
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
# CONFIGURACIÓN DE PÁGINA Y ESTILO VISUAL
# =====================================================
st.set_page_config(page_title="Clasificación de Aves", layout="centered", page_icon="🦜")
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #001f3f, #003366, #004080, #0a2342); color: #f0f8ff; }
h1, h2, h3, h4 { color: #33ccff; }
.stSelectbox label, .stFileUploader label { color: #e0f7fa; font-weight: 600; }
.stButton button { background-color: #0074D9; color: white; font-weight: bold; border-radius: 10px; border: none; transition: 0.3s; }
.stButton button:hover { background-color: #33ccff; color: #001f3f; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# BARRA LATERAL
# =====================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=90)
st.sidebar.title("🕊️ Sobre la aplicación")
st.sidebar.markdown("""
**Clasificación Automática de Aves**  
Esta aplicación usa modelos de **IA** para reconocer especies de aves a partir de imágenes.

### ⚙️ Funcionalidad:
- Subir una foto de un ave.
- Clasificar usando tres modelos: **VGG16**, **Xception**, **Cuántico Simulado**.
- Mostrar las **3 especies más probables** con su porcentaje de certeza.

### 🌱 Propósito:
Apoyar investigación, educación ambiental y conservación de biodiversidad.
""")

# =====================================================
# CABECERA
# =====================================================
st.title("🦜 Clasificación Automática de Aves")
st.write("Selecciona un modelo y sube una imagen para identificar la especie del ave.")

# =====================================================
# RUTAS Y LINK DE DRIVE
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")
ESPECIES_DIR = os.path.join(BASE_DIR, "especies_img")
os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(ESPECIES_DIR, exist_ok=True)

MODELOS = {
    "VGG16": {"id": "1XxRzz3sp_SDKFzxds7A3gFsVaNpkMcfr", "file": os.path.join(MODELOS_DIR, "vgg16_aves_final.keras")},
    "Xception": {"id": "1O-INGJMoeT84dGEq2sRrnoeBjCWKVIWn", "file": os.path.join(MODELOS_DIR, "xception_aves_final.keras")},
    "Cuántico Simulado": {"id": "1-qumvQ7c2Ipd5h-QF7rEYLRcoJMEcDpk", "file": os.path.join(MODELOS_DIR, "modelo_cuantico_simulado_aves_final.keras")}
}

# =====================================================
# FUNCIONES
# =====================================================
def descargar_modelo(nombre_modelo):
    path = MODELOS[nombre_modelo]["file"]
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={MODELOS[nombre_modelo]['id']}"
        gdown.download(url, path, quiet=False)
    return path

@st.cache_resource
def cargar_modelo(nombre_modelo):
    modelo_path = descargar_modelo(nombre_modelo)
    if "cuantico" in nombre_modelo.lower():
        return tf.keras.models.load_model(modelo_path, custom_objects={"SimulatedQuantumLayer": SimulatedQuantumLayer})
    else:
        return tf.keras.models.load_model(modelo_path)

def preparar_imagen(img, modelo_nombre):
    target_size = (299, 299) if "xception" in modelo_nombre.lower() else (224, 224)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# =====================================================
# INTERFAZ PRINCIPAL
# =====================================================
modelo_seleccionado = st.selectbox("Selecciona el modelo a utilizar:", list(MODELOS.keys()))
modelo = cargar_modelo(modelo_seleccionado)

uploaded_file = st.file_uploader("Sube una imagen del ave:", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Imagen cargada", use_container_width=True)

    if st.button("🔍 Clasificar"):
        with st.spinner("Analizando imagen..."):
            img_array = preparar_imagen(img, modelo_seleccionado)
            pred = modelo.predict(img_array)[0]

            clases = [
                "Accipiter bicolor", "Ardea cocoi", "Buteo albigula",
                "Cathartes burrovianus", "Chondrohierax uncinatus",
                "Dryocopus lineatus", "Egretta thula", "Falco columbarius",
                "Melanerpes formicivorus", "Sarcoramphus papa"
            ]

            top_indices = np.argsort(pred)[-3:][::-1]
            top_especies = [clases[i] for i in top_indices]
            top_probabilidades = [float(pred[i] * 100) for i in top_indices]
            especie_predicha = top_especies[0]

            st.success(f"🕊️ Especie predicha: **{especie_predicha}**")

            # =====================================================
            # GRÁFICO
            # =====================================================
            fig = px.bar(
                x=top_especies,
                y=top_probabilidades,
                text=[f"{p:.1f}%" for p in top_probabilidades],
                color=top_especies,
                color_discrete_sequence=["#1E90FF", "#00BFFF", "#87CEFA"]
            )
            fig.update_traces(
                textposition="outside",
                marker=dict(line=dict(color="#001f3f", width=1.5)),
                opacity=0.9
            )
            fig.update_layout(
                title="Top 3 especies más probables",
                title_x=0.5,
                yaxis_title="Probabilidad (%)",
                xaxis_title="Especies",
                template="plotly_dark",
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f0f8ff")
            )
            st.plotly_chart(fig, use_container_width=True)

            # =====================================================
            # IMAGEN DE REFERENCIA
            # =====================================================
            img_especie_path = os.path.join(ESPECIES_DIR, f"{especie_predicha}.jpg")
            if not os.path.exists(img_especie_path):
                # Descargar la carpeta comprimida de imágenes si no existe
                url = "https://drive.google.com/uc?id=1IWzvSSY-6oFeNmu3_DFgHM7snRXppqkU"
                zip_path = os.path.join(ESPECIES_DIR, "especies_img.zip")
                if not os.path.exists(zip_path):
                    gdown.download(url, zip_path, quiet=False)
                    import zipfile
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(ESPECIES_DIR)

            if os.path.exists(img_especie_path):
                st.image(img_especie_path, caption=f"Ejemplo de {especie_predicha}", use_container_width=True)
            else:
                st.warning("⚠️ No hay imagen de referencia disponible para esta especie.")
