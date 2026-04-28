# app.py - Entrenador Vocal con Streamlit y Hume AI
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from hume import HumeClient
import tempfile

# ------------------------------------------------------------
# Importación robusta de ProsodyConfig
# ------------------------------------------------------------
ProsodyConfig = None
try:
    from hume.expression_measurement import ProsodyConfig
except ImportError:
    try:
        from hume.models.config import ProsodyConfig
    except ImportError:
        try:
            from hume.expression_measurement.batch import ProsodyConfig
        except ImportError:
            st.error("No se pudo importar ProsodyConfig. Revisa la instalación de 'hume'.")
            st.stop()

# ------------------------------------------------------------
# Configuración inicial de la página
# ------------------------------------------------------------
st.set_page_config(page_title="Entrenador Vocal", page_icon="🎤", layout="wide")
st.title("🎤 Entrenador Vocal con Hume AI")
st.markdown("Analiza la entonación, ritmo y tonalidad de tu voz para sonar más **persuasiva**, **directa** o **experta**.")

# ------------------------------------------------------------
# Carga de API Key
# ------------------------------------------------------------
load_dotenv()
api_key = os.getenv("HUME_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("🔑 Ingresa tu API Key de Hume", type="password")
    if not api_key:
        st.sidebar.warning("Necesitas una API Key para usar la aplicación.")
        st.stop()

# ------------------------------------------------------------
# Perfiles ideales (los mismos de antes)
# ------------------------------------------------------------
IDEAL_PROFILES = {
    "persuasiva": {
        "Confidence": 0.85,
        "Excitement": 0.65,
        "Joy": 0.60,
        "Calmness": 0.50,
        "Determination": 0.75,
        "Doubt": 0.05,
        "Anxiety": 0.10,
        "Contemplation": 0.30
    },
    "directa": {
        "Confidence": 0.80,
        "Determination": 0.90,
        "Anger": 0.25,
        "Calmness": 0.60,
        "Doubt": 0.05,
        "Anxiety": 0.10,
        "Excitement": 0.40,
        "Contemplation": 0.20
    },
    "experta": {
        "Confidence": 0.90,
        "Calmness": 0.70,
        "Concentration": 0.60,
        "Contemplation": 0.50,
        "Doubt": 0.00,
        "Anxiety": 0.05,
        "Excitement": 0.30,
        "Determination": 0.70
    }
}

# ------------------------------------------------------------
# Función para extraer emociones (la misma versión robusta)
# ------------------------------------------------------------
def extract_emotion_scores(predictions):
    emotion_totals = {}
    count = 0
    for file_pred in predictions:
        # Intento 1: objetos con .models.prosody.grouped_predictions
        try:
            grouped = file_pred.models.prosody.grouped_predictions
            for group in grouped:
                for pred in group.predictions:
                    for emotion in pred.emotions:
                        name = emotion.name
                        emotion_totals[name] = emotion_totals.get(name, 0.0) + emotion.score
                        count += 1
            continue
        except AttributeError:
            pass
        # Intento 2: diccionario
        try:
            if isinstance(file_pred, dict):
                data = file_pred['results']['predictions'][0]['models']['prosody']['grouped_predictions']
            else:
                data = file_pred.results['predictions'][0]['models']['prosody']['grouped_predictions']
            for group in data:
                for seg in group.get('predictions', []):
                    for emotion in seg.get('emotions', []):
                        name = emotion.get('name') or emotion.name
                        score = emotion.get('score', 0) or emotion.score
                        emotion_totals[name] = emotion_totals.get(name, 0.0) + score
                        count += 1
            continue
        except (AttributeError, KeyError, TypeError):
            pass
        # Intento 3: estructura plana
        try:
            for segment in file_pred.predictions:
                if hasattr(segment, 'emotions'):
                    for emotion in segment.emotions:
                        name = emotion.name
                        emotion_totals[name] = emotion_totals.get(name, 0.0) + emotion.score
                        count += 1
                elif isinstance(segment, dict) and 'emotions' in segment:
                    for emotion in segment['emotions']:
                        name = emotion['name']
                        emotion_totals[name] = emotion_totals.get(name, 0.0) + emotion['score']
                        count += 1
            continue
        except AttributeError:
            pass
        st.warning(f"No se pudo extraer emociones de un segmento de tipo {type(file_pred)}")
    if count == 0:
        return {}
    return {name: total/count for name, total in emotion_totals.items()}

# ------------------------------------------------------------
# Función para generar feedback (igual a la original)
# ------------------------------------------------------------
def generar_feedback(scores, estilo):
    ideal = IDEAL_PROFILES[estilo]
    feedback = []
    radar_data = {}
    for emocion, valor_objetivo in ideal.items():
        actual = scores.get(emocion, 0.0)
        diff = valor_objetivo - actual
        radar_data[emocion] = {"actual": actual, "target": valor_objetivo, "diff": diff}
        if diff > 0.15:
            if emocion == "Confidence":
                feedback.append("🔴 **Confianza baja**: habla con más firmeza, evita terminar frases con tono ascendente (*uptalk*). Practica finales de frase descendentes.")
            elif emocion == "Excitement":
                feedback.append("🔴 **Poca energía**: varía más el volumen y la velocidad. Imagina que cuentas una historia emocionante.")
            elif emocion == "Determination":
                feedback.append("🔴 **Determinación baja**: enfatiza palabras clave con un leve aumento del volumen y un ritmo más pausado. Elimina muletillas como 'eh...'.")
            elif emocion == "Calmness":
                feedback.append("🔴 **Falta de calma (inseguridad)**: reduce la velocidad, haz pausas estratégicas y respira antes de empezar.")
        elif diff < -0.15:
            if emocion == "Anxiety":
                feedback.append("🟡 **Nerviosismo**: exhala lentamente antes de hablar y practica con un ritmo más relajado.")
            elif emocion == "Doubt":
                feedback.append("🟡 **Dudas vocales**: elimina vacilaciones, alarga ligeramente las vocales de palabras importantes.")
            elif emocion == "Anger" and estilo != "directa":
                feedback.append("🟡 **Tono agresivo**: baja el volumen y añade más pausas. La persuasión suave suele ser más eficaz.")
    if scores.get("Boredom", 0) > 0.4:
        feedback.append("⚪ **Voz monótona**: practica exagerar las subidas y bajadas de tono mientras grabas.")
    return feedback, radar_data

# ------------------------------------------------------------
# Gráfico de radar (devuelve la figura)
# ------------------------------------------------------------
def crear_radar(radar_data, estilo):
    etiquetas = list(radar_data.keys())
    actuales = [v["actual"] for v in radar_data.values()]
    objetivos = [v["target"] for v in radar_data.values()]
    angulos = np.linspace(0, 2 * np.pi, len(etiquetas), endpoint=False).tolist()
    actuales += actuales[:1]
    objetivos += objetivos[:1]
    angulos += angulos[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angulos, actuales, 'o-', linewidth=2, label='Tu voz')
    ax.fill(angulos, actuales, alpha=0.25)
    ax.plot(angulos, objetivos, 'o-', linewidth=2, label=f'Ideal {estilo}')
    ax.fill(angulos, objetivos, alpha=0.1)
    ax.set_thetagrids(np.degrees(angulos[:-1]), etiquetas)
    ax.set_ylim(0, 1)
    ax.set_title(f'Perfil vocal para estilo "{estilo}"')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    return fig

# ------------------------------------------------------------
# Interfaz principal de Streamlit
# ------------------------------------------------------------
st.sidebar.header("Configuración")
estilo = st.sidebar.selectbox(
    "¿Qué estilo quieres practicar?",
    ("persuasiva", "directa", "experta")
)

archivo_subido = st.file_uploader("Sube tu grabación de voz (formato WAV o MP3)", type=["wav", "mp3"])

if archivo_subido is not None:
    # Guardar el archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(archivo_subido.read())
        audio_path = tmp.name

    st.audio(archivo_subido, format='audio/wav')

    if st.button("Analizar mi voz"):
        with st.spinner("Analizando tu voz con Hume AI... Esto puede tardar unos segundos."):
            try:
                client = HumeClient(api_key=api_key)
                job = client.expression_measurement.batch.start_inference_job(
                    files=[audio_path],
                    models=[ProsodyConfig()]
                )
                result = job.await_complete()
                predictions = result.get_predictions()
                scores = extract_emotion_scores(predictions)

                if not scores:
                    st.error("No se pudieron extraer emociones del audio. Verifica que el archivo tenga voz clara.")
                else:
                    st.success("✅ Análisis completado")

                    # Mostrar scores
                    st.subheader("📊 Emociones detectadas en tu voz")
                    cols = st.columns(len(scores))
                    for i, (emo, val) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True)):
                        cols[i % 4].metric(label=emo, value=f"{val:.2f}")

                    # Recomendaciones
                    feedback, radar_data = generar_feedback(scores, estilo)
                    st.subheader(f"🎯 Recomendaciones para sonar más {estilo}")
                    for rec in feedback:
                        st.markdown(f"- {rec}")

                    # Gráfico radar
                    st.subheader("📈 Comparación visual")
                    fig = crear_radar(radar_data, estilo)
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Ocurrió un error durante el análisis: {str(e)}")
            finally:
                # Limpiar archivo temporal
                os.unlink(audio_path)
else:
    st.info("👆 Sube un archivo de audio para comenzar")