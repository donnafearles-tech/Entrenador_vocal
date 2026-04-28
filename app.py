# app.py - Entrenador Vocal con Hume AI (API REST)
import streamlit as st
import os
import requests
import time
import json
import tempfile
import plotly.graph_objects as go

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
    url_status = f"{HUME_BASE_URL}/{job_id}"
    url_predictions = f"{HUME_BASE_URL}/{job_id}/predictions"
    headers = {"X-Hume-Api-Key": api_key}
    
    while True:
        response = requests.get(url_status, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error al obtener estado: {response.status_code}")
        
        data = response.json()
        state = data.get("state") or data.get("status")
        if isinstance(state, dict):
            state = state.get("state") or state.get("status")
        if state is None and "job" in data:
            state = data["job"].get("state") or data["job"].get("status")
        if state is None:
            raise Exception(f"No se pudo determinar el estado. Respuesta: {data}")
            
        state = state.lower()
        
        if state == "completed":
            pred_response = requests.get(url_predictions, headers=headers)
            if pred_response.status_code == 200:
                return {"predictions": pred_response.json()}
            else:
                raise Exception(f"Error al obtener predicciones: {pred_response.status_code}")
                
        elif state in ("failed", "cancelled"):
            raise Exception(f"El trabajo de Hume terminó con error o fue cancelado. Estado: {state}")
        else:
            time.sleep(2)

def extract_emotion_scores(predictions_payload):
    emotion_totals = {}
    segment_count = 0  
    
    for item in predictions_payload:
        if "results" in item and "predictions" in item["results"]:
            for pred in item["results"]["predictions"]:
                if "models" in pred and "prosody" in pred["models"]:
                    prosody = pred["models"]["prosody"]
                    if "grouped_predictions" in prosody:
                        for group in prosody["grouped_predictions"]:
                            for segment in group.get("predictions", []):
                                segment_count += 1  
                                for emo in segment.get("emotions", []):
                                    name = emo.get("name", "")
                                    score = emo.get("score", 0.0)
                                    emotion_totals[name] = emotion_totals.get(name, 0.0) + score
                                    
    if segment_count == 0:
        return {}
        
    return {name: total / segment_count for name, total in emotion_totals.items()}

# ------------------------------------------------------------
# Feedback y Visualización (Plotly y Kanban)
# ------------------------------------------------------------
def generar_feedback(scores, estilo):
    ideal = IDEAL_PROFILES[estilo]
    feedback = []
    radar_data = {}
    for emocion, valor_objetivo in ideal.items():
        actual = scores.get(emocion, 0.0)
        diff = valor_objetivo - actual
        radar_data[emocion] = {"actual": actual, "target": valor_objetivo, "diff": diff}
        
        # Lógica de feedback basada en diferencias
        if diff > 0.15:
            if emocion == "Confidence":
                feedback.append("🔴 **Confianza baja**: habla con más firmeza, evita terminar frases con tono ascendente (*uptalk*).")
            elif emocion == "Excitement":
                feedback.append("🔴 **Poca energía**: varía más el volumen y la velocidad.")
            elif emocion == "Determination":
                feedback.append("🔴 **Determinación baja**: enfatiza palabras clave con un leve aumento del volumen.")
            elif emocion == "Calmness":
                feedback.append("🔴 **Falta de calma**: reduce la velocidad y respira antes de empezar.")
        elif diff < -0.15:
            if emocion == "Anxiety":
                feedback.append("🟡 **Nerviosismo**: exhala lentamente antes de hablar.")
            elif emocion == "Doubt":
                feedback.append("🟡 **Dudas vocales**: elimina vacilaciones y muletillas.")
            elif emocion == "Anger" and estilo != "directa":
                feedback.append("🟡 **Tono agresivo**: baja el volumen y añade más pausas.")
                
    if scores.get("Boredom", 0) > 0.4:
        feedback.append("⚪ **Voz monótona**: practica exagerar las subidas y bajadas de tono.")
        
    return feedback, radar_data

def crear_radar_plotly(radar_data, estilo):
    etiquetas = list(radar_data.keys())
    # Convertimos a porcentajes para el radar
    actuales = [v["actual"] * 100 for v in radar_data.values()]
    objetivos = [v["target"] * 100 for v in radar_data.values()]

    # Cerramos el polígono repitiendo el primer valor al final
    etiquetas.append(etiquetas[0])
    actuales.append(actuales[0])
    objetivos.append(objetivos[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=actuales,
        theta=etiquetas,
        fill='toself',
        name='Tu Voz',
        line_color='blue',
        text=[f"{v:.1f}%" for v in actuales],
        hoverinfo="text+name"
    ))

    fig.add_trace(go.Scatterpolar(
        r=objetivos,
        theta=etiquetas,
        fill='toself',
        name=f'Ideal: {estilo}',
        line_color='rgba(255, 165, 0, 0.7)',
        text=[f"{v:.1f}%" for v in objetivos],
        hoverinfo="text+name"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%"
            )
        ),
        showlegend=True,
        title=f'Perfil vocal interactivo - "{estilo.capitalize()}"'
    )
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
        with st.spinner("Analizando tu voz con Hume AI..."):
            try:
                job_id = start_job(api_key, audio_path)
                results = get_job_result(api_key, job_id)
                predictions = results.get("predictions", [])
                scores = extract_emotion_scores(predictions)

                if not scores:
                    st.error("No se pudieron extraer emociones del audio.")
                else:
                    st.success("✅ Análisis completado con éxito")
                    
                    st.markdown("---")
                    
                    # --------------------------------------------------------
                    # SECCIÓN KANBAN DE EMOCIONES
                    # --------------------------------------------------------
                    st.subheader("📋 Tablero Kanban de Emociones")
                    st.markdown("Clasificación de tu tono de voz según la intensidad detectada.")
                    
                    # Ordenar emociones de mayor a menor
                    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### 🔴 Alta Intensidad\n*(>25%)*")
                        for emo, val in sorted_emotions:
                            if val >= 0.25:
                                with st.container(border=True):
                                    st.markdown(f"**{emo}**")
                                    st.markdown(f"### {val*100:.1f}%")

                    with col2:
                        st.markdown("### 🟡 Media Intensidad\n*(10% - 25%)*")
                        for emo, val in sorted_emotions:
                            if 0.10 <= val < 0.25:
                                with st.container(border=True):
                                    st.markdown(f"**{emo}**")
                                    st.markdown(f"### {val*100:.1f}%")

                    with col3:
                        st.markdown("### ⚪ Baja Intensidad\n*(<10%)*")
                        # Solo mostramos las 5 más altas de este rango para no saturar la pantalla
                        count = 0
                        for emo, val in sorted_emotions:
                            if val < 0.10 and count < 5:
                                with st.container(border=True):
                                    st.markdown(f"**{emo}**")
                                    st.markdown(f"### {val*100:.1f}%")
                                count += 1
                        if len([e for e, v in sorted_emotions if v < 0.10]) > 5:
                            st.caption("*... y otras emociones menores.*")
                            
                    st.markdown("---")

                    # --------------------------------------------------------
                    # SECCIÓN RADAR INTERACTIVO
                    # --------------------------------------------------------
                    st.subheader("📈 Radar Comparativo")
                    feedback, radar_data = generar_feedback(scores, estilo)
                    
                    # Mostrar el gráfico de Plotly
                    fig = crear_radar_plotly(radar_data, estilo)
                    st.plotly_chart(fig, use_container_width=True)

                    # --------------------------------------------------------
                    # SECCIÓN FEEDBACK
                    # --------------------------------------------------------
                    st.subheader(f"🎯 Recomendaciones para sonar más {estilo}")
                    if len(feedback) == 0:
                        st.info("¡Excelente trabajo! Tu voz se ajusta muy bien al perfil seleccionado.")
                    else:
                        for rec in feedback:
                            st.markdown(f"- {rec}")

            except Exception as e:
                st.error(f"Ocurrió un error durante el análisis: {str(e)}")
            finally:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
else:
    st.info("👆 Sube un archivo de audio para comenzar")
