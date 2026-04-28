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
st.markdown("Analiza tu voz para sonar más **persuasiva**, **directa** o **experta**.")

# ------------------------------------------------------------
# Diccionario de Traducción de Emociones
# ------------------------------------------------------------
TRADUCCION_EMOCIONES = {
    "Admiration": "Admiración", "Adoration": "Adoración", "Aesthetic Appreciation": "Aprecio Estético",
    "Amusement": "Diversión", "Anger": "Enojo", "Anxiety": "Ansiedad", "Awe": "Asombro",
    "Awkwardness": "Incomodidad", "Boredom": "Aburrimiento", "Calmness": "Calma",
    "Concentration": "Concentración", "Confusion": "Confusión", "Contemplation": "Contemplación",
    "Contempt": "Desprecio", "Contentment": "Satisfacción", "Craving": "Anhelo",
    "Desire": "Deseo", "Determination": "Determinación", "Disappointment": "Decepción",
    "Disgust": "Asco", "Distress": "Angustia", "Doubt": "Duda", "Ecstasy": "Éxtasis",
    "Embarrassment": "Vergüenza", "Empathic Pain": "Dolor Empático", "Entrancement": "Arrobamiento",
    "Envy": "Envidia", "Excitement": "Entusiasmo", "Fear": "Miedo", "Guilt": "Culpa",
    "Horror": "Horror", "Interest": "Interés", "Joy": "Alegría", "Love": "Amor",
    "Nostalgia": "Nostalgia", "Pain": "Dolor", "Pride": "Orgullo", "Realization": "Darse Cuenta",
    "Relief": "Alivio", "Romance": "Romance", "Sadness": "Tristeza", "Satisfaction": "Placer",
    "Shame": "Vergüenza Ajena", "Surprise (negative)": "Sorpresa (negativa)",
    "Surprise (positive)": "Sorpresa (positiva)", "Sympathy": "Simpatía",
    "Tiredness": "Cansancio", "Triumph": "Triunfo", "Confidence": "Confianza"
}

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
        "Confianza": 0.85, "Entusiasmo": 0.65, "Alegría": 0.60,
        "Calma": 0.50, "Determinación": 0.75, "Duda": 0.05,
        "Ansiedad": 0.10, "Contemplación": 0.30
    },
    "directa": {
        "Confianza": 0.80, "Determinación": 0.90, "Enojo": 0.25,
        "Calma": 0.60, "Duda": 0.05, "Ansiedad": 0.10,
        "Entusiasmo": 0.40, "Contemplación": 0.20
    },
    "experta": {
        "Confianza": 0.90, "Calma": 0.70, "Concentración": 0.60,
        "Contemplación": 0.50, "Duda": 0.00, "Ansiedad": 0.05,
        "Entusiasmo": 0.30, "Determinación": 0.70
    }
}

# ------------------------------------------------------------
# Funciones API
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
        res = requests.get(url_status, headers=headers)
        data = res.json()
        
        # Extracción segura del estado
        state = data.get("state") or data.get("status")
        if isinstance(state, dict):
            state = state.get("status") or state.get("state", "")
            
        if state is None and "job" in data:
            state = data["job"].get("state") or data["job"].get("status", "")
            
        # Convertimos a string de forma segura antes de minúsculas
        state = str(state).lower()
        
        if state == "completed":
            pred_res = requests.get(url_predictions, headers=headers)
            return {"predictions": pred_res.json()}
        elif state in ("failed", "cancelled"):
            raise Exception(f"Error en el Job: {state}")
            
        time.sleep(2)

def extract_emotion_scores(predictions_payload):
    emotion_totals = {}
    segment_count = 0  
    for item in predictions_payload:
        if "results" in item and "predictions" in item["results"]:
            for pred in item["results"]["predictions"]:
                if "models" in pred and "prosody" in pred["models"]:
                    for group in pred["models"]["prosody"].get("grouped_predictions", []):
                        for segment in group.get("predictions", []):
                            segment_count += 1
                            for emo in segment.get("emotions", []):
                                eng_name = emo.get("name", "")
                                esp_name = TRADUCCION_EMOCIONES.get(eng_name, eng_name)
                                score = emo.get("score", 0.0)
                                emotion_totals[esp_name] = emotion_totals.get(esp_name, 0.0) + score
    if segment_count == 0: return {}
    return {name: total / segment_count for name, total in emotion_totals.items()}

# ------------------------------------------------------------
# Visualización y Feedback
# ------------------------------------------------------------
def crear_radar_plotly(radar_data, estilo):
    etiquetas = list(radar_data.keys())
    actuales = [v["actual"] * 100 for v in radar_data.values()]
    objetivos = [v["target"] * 100 for v in radar_data.values()]

    etiquetas.append(etiquetas[0])
    actuales.append(actuales[0])
    objetivos.append(objetivos[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=actuales, theta=etiquetas, fill='toself', name='Tu Voz', line_color='#1f77b4'))
    fig.add_trace(go.Scatterpolar(r=objetivos, theta=etiquetas, fill='toself', name=f'Objetivo: {estilo.capitalize()}', line_color='#ff7f0e'))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%", gridcolor="lightgrey"),
            angularaxis=dict(tickfont=dict(size=10), rotation=90, direction="clockwise")
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25, 
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=40, b=80), 
        height=500,
        title=dict(text="Comparativa de Tono", x=0.5, xanchor='center')
    )
    return fig

# ------------------------------------------------------------
# UI Streamlit
# ------------------------------------------------------------
st.sidebar.header("Opciones")
estilo = st.sidebar.selectbox("¿Qué estilo quieres practicar?", ("persuasiva", "directa", "experta"))

archivo_subido = st.file_uploader("Sube tu voz (WAV/MP3)", type=["wav", "mp3"])

if archivo_subido:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(archivo_subido.read())
        audio_path = tmp.name

    st.audio(archivo_subido)

    if st.button("Analizar ahora"):
        with st.spinner("Analizando matices vocales..."):
            try:
                job_id = start_job(api_key, audio_path)
                results = get_job_result(api_key, job_id)
                scores = extract_emotion_scores(results.get("predictions", []))

                if not scores:
                    st.error("No se detectó audio claro.")
                else:
                    st.success("✅ Análisis completado")

                    # Kanban
                    st.subheader("📋 Tablero de Intensidad Vocal")
                    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.markdown("### 🔴 Alta")
                        for e, v in sorted_emotions:
                            if v >= 0.25:
                                with st.container(border=True): st.markdown(f"**{e}**\n### {v*100:.1f}%")
                    with c2:
                        st.markdown("### 🟡 Media")
                        for e, v in sorted_emotions:
                            if 0.10 <= v < 0.25:
                                with st.container(border=True): st.markdown(f"**{e}**\n### {v*100:.1f}%")
                    with c3:
                        st.markdown("### ⚪ Baja")
                        for e, v in sorted_emotions[:15]: 
                            if v < 0.10:
                                with st.container(border=True): st.markdown(f"**{e}**\n### {v*100:.1f}%")

                    # Radar Centrado
                    st.markdown("---")
                    st.subheader("📈 Mapa de Perfil Vocal")
                    radar_data = {e: {"actual": scores.get(e, 0.0), "target": v} for e, v in IDEAL_PROFILES[estilo].items()}
                    fig = crear_radar_plotly(radar_data, estilo)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(audio_path): os.unlink(audio_path)
    
