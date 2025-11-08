import os
import streamlit as st
from PIL import Image
import pandas as pd
import tensorflow as tf
import gdown
from utils import predict

# ==========================
# 🎨 CONFIGURACIÓN BÁSICA
# ==========================
st.set_page_config(
    page_title="Clasificador Cuántico de Aves Colombianas",
    page_icon="🦅",
    layout="wide",
)

# Estilo visual personalizado (color de fondo, textos, botones)
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f5f5;
        color: #002b36;
    }
    .stButton>button {
        background-color: #004d4d;
        color: white;
        border-radius: 10px;
        height: 2.5em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #007777;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# 📂 MODELOS DESDE GOOGLE DRIVE
# ==========================
MODELS_DIR = "models"

# ✅ IDs de tus modelos reales en Google Drive
XCEPTION_ID = "1rOSSNrFkSNMpPil16qYMTEVJgu2PLJx8"
VGG16_ID = "1CtUBQxsPkwo89vr4fjbsp54gzJOus0xZ"

def descargar_modelo(file_id, nombre_local):
    """
    Descarga el modelo desde Google Drive a la carpeta models/
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    ruta_local = os.path.join(MODELS_DIR, nombre_local)
    if not os.path.exists(ruta_local):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"📥 Descargando {nombre_local}..."):
            gdown.download(url, ruta_local, quiet=False)
    return ruta_local

@st.cache_resource
def load_selected_model(model_name: str):
    """
    Carga el modelo Xception o VGG16 según elección del usuario.
    """
    try:
        if model_name == "Xception":
            path = descargar_modelo(XCEPTION_ID, "modelo_xception.keras")
        else:
            path = descargar_modelo(VGG16_ID, "modelo_vgg16.keras")

        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ No se pudo cargar el modelo {model_name}.")
        st.exception(e)
        st.stop()

# ==========================
# 🐦 INFO DE TUS ESPECIES
# ==========================
BIRD_INFO = {
    "Accipiter_bicolor": {
        "common": "Gavilán bicolor",
        "scientific": "Accipiter bicolor",
        "description": "Ave rapaz de tamaño mediano con partes superiores oscuras y partes inferiores blancas con rayas finas."
    },
    "Ardea_cocoi": {
        "common": "Garza mora",
        "scientific": "Ardea cocoi",
        "description": "Gran garza de plumaje gris y blanco, muy común en humedales y orillas de ríos."
    },
    "Buteo_albigula": {
        "common": "Aguililla pechiblanca",
        "scientific": "Buteo albigula",
        "description": "Ave rapaz de montaña con pecho blanco y alas anchas. Caza pequeños mamíferos y aves."
    },
    "Cathartes_burrovianus": {
        "common": "Gallinazo sabanero",
        "scientific": "Cathartes burrovianus",
        "description": "Zopilote de sabana con cabeza desnuda y amarillenta, conocido por su vuelo bajo en áreas abiertas."
    },
    "Chondrohierax_uncinatus": {
        "common": "Gavilán caracolero",
        "scientific": "Chondrohierax uncinatus",
        "description": "Especialista en alimentarse de caracoles; tiene un pico curvado ideal para extraerlos."
    },
    "Dryocopus_lineatus": {
        "common": "Carpintero lineado",
        "scientific": "Dryocopus lineatus",
        "description": "Carpintero grande de color negro con una llamativa cresta roja. Golpetea árboles en busca de insectos."
    },
    "Egretta_thula": {
        "common": "Garceta nívea",
        "scientific": "Egretta thula",
        "description": "Garza blanca elegante con patas negras y pies amarillos, habitual en lagunas y manglares."
    },
    "Falco_columbarius": {
        "common": "Halcón esmerejón",
        "scientific": "Falco columbarius",
        "description": "Pequeño halcón cazador de vuelo rápido, se alimenta de aves pequeñas y es muy ágil."
    },
    "Melanerpes_formicivorus": {
        "common": "Carpintero bellotero",
        "scientific": "Melanerpes formicivorus",
        "description": "Carpintero social que almacena bellotas en huecos de árboles; común en bosques templados."
    },
    "Sarcoramphus_papa": {
        "common": "Zopilote rey",
        "scientific": "Sarcoramphus papa",
        "description": "Imponente buitre de cabeza multicolor y gran tamaño, símbolo de los bosques tropicales."
    },
}

# ==========================
# 🎛 SIDEBAR
# ==========================
with st.sidebar:
    st.title("🦅 Clasificador de Aves Cuántico")
    st.markdown(
        "Selecciona el modelo de deep learning con el que deseas analizar tus imágenes de aves."
    )

    model_name = st.selectbox(
        "📘 Modelo de clasificación",
        ["Xception", "VGG16"],
        help="Puedes probar y comparar los resultados entre ambos modelos."
    )

    st.markdown("### 🐦 Especies disponibles")
    for key, info in BIRD_INFO.items():
        st.markdown(
            f"- **{info['common']}**  \n"
            f"  <span style='font-size:12px;'>{info['scientific']}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption("💡 Consejo: usa imágenes nítidas, centradas y con buena iluminación para mejores resultados.")

# Cargar modelo seleccionado
model = load_selected_model(model_name)

# ==========================
# 🖼 INTERFAZ PRINCIPAL
# ==========================
st.markdown("## 📸 Clasifica tu imagen de ave")

col_left, col_right = st.columns([1.2, 1])

uploaded_file = col_left.file_uploader(
    "Sube una imagen (JPG o PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    col_left.image(img, caption="📷 Imagen cargada", use_column_width=True)

    if col_left.button("🔍 Analizar imagen"):
        with st.spinner(f"Ejecutando modelo {model_name}..."):
            results = predict(model, img, model_type=model_name)

        if not results:
            st.error("❌ No se pudieron obtener predicciones. Verifica la función predict() en utils.py.")
        else:
            results = sorted(results, key=lambda x: x[1], reverse=True)
            best_name, best_prob = results[0]

            best_info = BIRD_INFO.get(
                best_name,
                {"common": best_name, "scientific": best_name, "description": "Sin descripción disponible."}
            )

            with col_right:
                st.markdown("### ✅ Resultado principal")
                st.success(
                    f"Es muy probable que sea **{best_info['common']}** "
                    f"(*{best_info['scientific']}*)\n\n"
                    f"Confianza del modelo: **{best_prob*100:.2f}%**"
                )
                st.markdown("#### 📝 Descripción")
                st.write(best_info["description"])

            st.markdown("### 📊 Predicciones del modelo")
            labels = []
            probs = []
            for name, prob in results:
                info = BIRD_INFO.get(name, {"common": name})
                labels.append(info["common"])
                probs.append(prob * 100)

            df = pd.DataFrame({"Especie": labels, "Probabilidad (%)": probs}).set_index("Especie")
            st.bar_chart(df)

else:
    col_right.info("👈 Sube una imagen a la izquierda para ver aquí la predicción y la descripción del ave.")
