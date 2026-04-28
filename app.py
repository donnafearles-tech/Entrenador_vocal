# app.py - Entrenador Vocal con Hume AI (API REST)
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import json
import tempfile

# ------------------------------------------------------------
# Configuración de la página
# ------------------------------------------------------------
st.set_page_config(page_title="Entrenador Vocal", page_icon="🎤", layout="wide")
st.title("🎤 Entrenador Vocal con Hume AI")
st.markdown("Analiza la entonación, ritmo y tonalidad de tu voz para sonar más **persuasiva**, **directa** o **experta**.")

# ------------------------------------------------------------
# Obtener API Key
# ------------------------------------------------------------
try:
    api_key = st.secrets["HUME_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("HUME_API_KEY", "")

if not api_key:
    st.error("🔑 No se encontró la API Key de Hume.")
    st.stop()

# ------------------------------------------------------------
# Perfiles ideales
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
# Funciones para la API REST de Hume
# ------------------------------------------------------------
HUME_BASE_URL = "https://api.hume.ai/v0/batch/jobs"

def start_job(api_key, file_path):
    headers = {"X-Hume-Api-Key": api_key}
    with open(file_path, "rb") as f:
        files = {"file": f}
        json_payload = json.dumps({"models": {"prosody": {}}})
        data = {"json": json_payload}
        response = requests.post(HUME_BASE_URL, files=files, data=data, headers=headers)
    if response.status_code == 200:
        return response.json()["job_id"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def get_job_result(api_key, job_id):
    url = f"{HUME_BASE_URL}/{job_id}"
    headers = {"X-Hume-Api-Key": api_key}
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error al obtener estado: {response.status_code}")
        data = response.json()
        # Obtener estado
        state = data.get("state") or data.get("status")
        if isinstance(state, dict):
            state = state.get("state") or state.get("status")
        if state is None and "job" in data:
            state = data["job"].get("state") or data["job"].get("status")
        if state is None:
            raise Exception(f"No se pudo determinar el estado. Respuesta: {data}")
        state = state.lower()
        if state == "completed":
            return data.get("results", {})
        elif state in ("failed", "cancelled"):
            raise Exception(f"Job {state}")
        else:
            time.sleep(2)

def extract_emotion_scores(predictions_payload):
    """Extrae puntuaciones promedio. Soporta varias estructuras."""
    emotion_totals = {}
    count = 0
    for file_result in predictions_payload:
        # Intento 1: file_result["models"]["prosody"]["grouped_predictions"]
        if "models" in file_result and "prosody" in file_result["models"]:
            prosody = file_result["models"]["prosody"]
            if "grouped_predictions" in prosody:
                segments = []
                for group in prosody["grouped_predictions"]:
                    segments.extend(group.get("predictions", []))
            elif "predictions" in prosody:
                segments = prosody["predictions"]
            else:
                continue
            for seg in segments:
                for emo in seg.get("emotions", []):
                    name = emo.get("name", "")
                    score = emo.get("score", 0.0)
                    emotion_totals[name] = emotion_totals.get(name, 0.0) + score
                    count += 1
        # Intento 2: estructura plana con "emotions" en el nivel superior
        elif "emotions" in file_result:
            for emo in file_result["emotions"]:
                name = emo.get("name", "")
                score = emo.get("score", 0.0)
                emotion_totals[name] = emotion_totals.get(name, 0.0) + score
                count += 1
        # Intento 3: a veces las predicciones están dentro de "predictions" anidadas
        elif "predictions" in file_result:
            for pred in file_result["predictions"]:
                if isinstance(pred, dict):
                    if "emotions" in pred:
                        for emo in pred["emotions"]:
                            name = emo.get("name", "")
                            score = emo.get("score", 0.0)
                            emotion_totals[name] = emotion_totals.get(name, 0.0) + score
                            count += 1
    if count == 0:
        return {}
    return {name: total/count for name, total in emotion_totals.items()}

# ------------------------------------------------------------
# Feedback y radar
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
# Interfaz de Streamlit
# ------------------------------------------------------------
st.sidebar.header("Configuración")
estilo = st.sidebar.selectbox(
    "¿Qué estilo quieres practicar?",
    ("persuasiva", "directa", "experta")
)

archivo_subido = st.file_uploader("Sube tu grabación de voz (formato WAV o MP3)", type=["wav", "mp3"])

if archivo_subido is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(archivo_subido.read())
        audio_path = tmp.name

    st.audio(archivo_subido, format='audio/wav')

    if st.button("Analizar mi voz"):
        with st.spinner("Analizando tu voz con Hume AI... Esto puede tardar unos segundos."):
            try:
                # 1. Iniciar job
                job_id = start_job(api_key, audio_path)
                st.text(f"Job ID: {job_id} (procesando...)")

                # 2. Obtener resultados completos
                results = get_job_result(api_key, job_id)

                # DEBUG: mostrar todo el objeto results (la clave está aquí)
                st.write("🧪 **Objeto 'results' completo:**")
                st.json(results)

                # Extraemos las predicciones reales de la estructura correcta.
                # En algunos casos la lista de archivos procesados está en "predictions",
                # pero si está vacía puede estar en "files" o dentro de "predictions" anidadas.
                predictions = results.get("predictions", [])
                if not predictions:
                    # Intentar otras rutas
                    if "files" in results:
                        predictions = results["files"]
                    elif "data" in results:
                        predictions = results["data"]
                    elif isinstance(results, list) and len(results) > 0:
                        # Por si results es directamente la lista
                        predictions = results

                scores = extract_emotion_scores(predictions)

                if not scores:
                    st.error("No se pudieron extraer emociones del audio. Revisa el JSON de arriba para ver la estructura.")
                else:
                    st.success("✅ Análisis completado")

                    st.subheader("📊 Emociones detectadas en tu voz")
                    cols = st.columns(len(scores))
                    for i, (emo, val) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True)):
                        cols[i % 4].metric(label=emo, value=f"{val:.2f}")

                    feedback, radar_data = generar_feedback(scores, estilo)
                    st.subheader(f"🎯 Recomendaciones para sonar más {estilo}")
                    for rec in feedback:
                        st.markdown(f"- {rec}")

                    st.subheader("📈 Comparación visual")
                    fig = crear_radar(radar_data, estilo)
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Ocurrió un error durante el análisis: {str(e)}")
            finally:
                os.unlink(audio_path)
else:
    st.info("👆 Sube un archivo de audio para comenzar")
