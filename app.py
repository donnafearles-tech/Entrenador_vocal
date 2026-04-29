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
        json_payload = json.dumps({"models": {"prosody": {}, "language": {}}})
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
        
        state = data.get("state") or data.get("status")
        if isinstance(state, dict):
            state = state.get("status") or state.get("state", "")
            
        if state is None and "job" in data:
            state = data["job"].get("state") or data["job"].get("status", "")
            
        state = str(state).lower()
        
        if state == "completed":
            pred_res = requests.get(url_predictions, headers=headers)
            return {"predictions": pred_res.json()}
        elif state in ("failed", "cancelled"):
            raise Exception(f"Error en el Job: {state}")
            
        time.sleep(2)

def calcular_confianza_artificial(scores):
    """Calcula la confianza en base a una fórmula ponderada."""
    determinacion = scores.get("Determinación", 0.0)
    calma = scores.get("Calma", 0.0)
    entusiasmo = scores.get("Entusiasmo", 0.0)
    ansiedad = scores.get("Ansiedad", 0.0)
    duda = scores.get("Duda", 0.0)

    confianza = (determinacion * 0.4) + (calma * 0.3) + (entusiasmo * 0.2) - (ansiedad * 0.5) - (duda * 0.5)
    
    # Normalizar para que el valor esté entre 0 y 1
    confianza_normalizada = max(0.0, min(1.0, confianza))
    return confianza_normalizada

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
    
    if segment_count == 0:
        return {}
        
    promedios = {name: total / segment_count for name, total in emotion_totals.items()}
    
    # 🌟 Integración de la nueva función de Confianza
    promedios["Confianza"] = calcular_confianza_artificial(promedios)
    
    return promedios

# ------------------------------------------------------------
# Visualización y Feedback
# ------------------------------------------------------------
def generar_feedback(scores, estilo):
    ideal = IDEAL_PROFILES[estilo]
    feedback = []
    
    for emocion, valor_objetivo in ideal.items():
        actual = scores.get(emocion, 0.0)
        diff = valor_objetivo - actual
        
        if diff > 0.15:
            if emocion == "Confianza":
                feedback.append("🔴 **Confianza baja**: habla con más firmeza, evita terminar frases con tono ascendente (*uptalk*).")
            elif emocion == "Entusiasmo":
                feedback.append("🔴 **Poca energía**: varía más el volumen y la velocidad. Imagina que cuentas una historia.")
            elif emocion == "Determinación":
                feedback.append("🔴 **Determinación baja**: enfatiza palabras clave con un leve aumento del volumen.")
            elif emocion == "Calma":
                feedback.append("🔴 **Falta de calma (inseguridad)**: reduce la velocidad, haz pausas estratégicas y respira antes de empezar.")
        elif diff < -0.15:
            if emocion == "Ansiedad":
                feedback.append("🟡 **Nerviosismo**: exhala lentamente antes de hablar y practica con un ritmo más relajado.")
            elif emocion == "Duda":
                feedback.append("🟡 **Dudas vocales**: elimina vacilaciones y muletillas.")
            elif emocion == "Enojo" and estilo != "directa":
                feedback.append("🟡 **Tono agresivo**: baja el volumen y añade más pausas. La persuasión suave suele ser más eficaz.")
                
    if scores.get("Aburrimiento", 0) > 0.4:
        feedback.append("⚪ **Voz monótona**: practica exagerar las subidas y bajadas de tono mientras grabas.")
        
    return feedback

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
        title=dict(text="Comparativa de Tono", x=0.5, xanchor='center'),
        dragmode="zoom",
        hovermode="closest"
    )
    
    fig.update_xaxes(showspikes=True, spikemode='across')
    fig.update_yaxes(showspikes=True, spikemode='across')
    
    return fig

# ------------------------------------------------------------
# UI Streamlit
# ------------------------------------------------------------
st.sidebar.header("Opciones")
estilo = st.sidebar.selectbox("¿Qué estilo quieres practicar?", ("persuasiva", "directa", "experta"))

archivo_subido = st.file_uploader("Sube tu voz (WAV/MP3)", type=["wav", "mp3"])

if archivo_subido:
    _, extension = os.path.splitext(archivo_subido.name)
    if not extension:
        extension = ".wav"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(archivo_subido.read())
        audio_path = tmp.name

    st.audio(archivo_subido)
    
    tamano_mb = len(archivo_subido.getvalue()) / (1024 * 1024)
    st.write(f"📦 Tamaño del archivo: {tamano_mb:.2f} MB")
    
    if st.button("Analizar ahora"):
        with st.spinner("Analizando matices vocales y transcribiendo..."):
            try:
                job_id = start_job(api_key, audio_path)
                results = get_job_result(api_key, job_id)
                predictions = results.get("predictions", [])
                scores = extract_emotion_scores(predictions)

                if not scores:
                    st.error("No se detectó audio claro.")
                else:
                    st.success("✅ Análisis completado")
                    
                    if "Confianza" in scores:
                        st.info(f"📊 **Nivel de Confianza proyectada:** {scores['Confianza']*100:.2f}%")

                    # Transcripción
                    textos_transcritos = []
                    try:
                        for item in predictions:
                            if "results" in item and "predictions" in item["results"]:
                                for pred in item["results"]["predictions"]:
                                    if "models" in pred and "language" in pred["models"]:
                                        for group in pred["models"]["language"].get("grouped_predictions", []):
                                            for segment in group.get("predictions", []):
                                                texto = segment.get("text", "").strip()
                                                if texto:
                                                    textos_transcritos.append(texto)
                        
                        texto_completo = " ".join(textos_transcritos)
                        if texto_completo:
                            st.info(f"📝 **Texto completo transcrito:** \"{texto_completo}\"")
                        else:
                            st.warning("⚠️ No se detectaron palabras claras para transcribir.")
                    except Exception as e:
                        st.warning(f"Error al extraer la transcripción: {e}")

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

                    # Radar
                    st.markdown("---")
                    st.subheader("📈 Mapa de Perfil Vocal")
                    st.markdown("💡 *Puedes hacer zoom en el gráfico: arrastra para zoom, usa la rueda del ratón o los botones de control.*")
                    radar_data = {e: {"actual": scores.get(e, 0.0), "target": v} for e, v in IDEAL_PROFILES[estilo].items()}
                    fig = crear_radar_plotly(radar_data, estilo)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

                    # Recomendaciones
                    st.markdown("---")
                    st.subheader(f"🎯 Tips para sonar más {estilo}")
                    feedback_textos = generar_feedback(scores, estilo)
                    
                    if not feedback_textos:
                        st.info("¡Excelente trabajo! Tu voz se ajusta muy bien al perfil seleccionado.")
                    else:
                        for rec in feedback_textos:
                            st.markdown(f"- {rec}")

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(audio_path): os.unlink(audio_path)
        
