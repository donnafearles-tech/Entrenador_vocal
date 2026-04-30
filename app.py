# app.py - English Vocal Coach Pro
import streamlit as st
import os
import requests
import time
import json
import tempfile
import plotly.graph_objects as go
import eng_to_ipa as ipa
import librosa
import numpy as np
import gc

# ------------------------------------------------------------
# Configuración de la página
# ------------------------------------------------------------
st.set_page_config(page_title="English Vocal Coach Pro", page_icon="🎤", layout="wide")
st.title("🎤 English Vocal Coach Pro")
st.markdown("""
Analiza tu pronunciación y tono en inglés. 
Esta herramienta compara tu **énfasis físico (energía)** con el **estándar fonético (IPA)**.
""")

# ------------------------------------------------------------
# Diccionarios y Perfiles
# ------------------------------------------------------------
TRADUCCION_EMOCIONES = {
    "Admiration": "Admiration", "Anger": "Anger", "Anxiety": "Anxiety", "Awe": "Awe",
    "Awkwardness": "Awkwardness", "Boredom": "Boredom", "Calmness": "Calmness",
    "Concentration": "Concentration", "Confusion": "Confusion", "Contempt": "Contempt",
    "Contentment": "Contentment", "Determination": "Determination", "Doubt": "Doubt",
    "Excitement": "Excitement", "Fear": "Fear", "Interest": "Interest", "Joy": "Joy",
    "Pride": "Pride", "Sadness": "Sadness", "Surprise (positive)": "Surprise (+)",
    "Tiredness": "Tiredness", "Triumph": "Triumph"
}

IDEAL_PROFILES = {
    "persuasive": {
        "Confidence": 0.85, "Excitement": 0.65, "Joy": 0.60,
        "Calmness": 0.50, "Determination": 0.75, "Doubt": 0.05,
        "Anxiety": 0.10, "Concentration": 0.30
    },
    "direct": {
        "Confidence": 0.80, "Determination": 0.90, "Anger": 0.25,
        "Calmness": 0.60, "Doubt": 0.05, "Anxiety": 0.10,
        "Excitement": 0.40, "Concentration": 0.20
    },
    "expert": {
        "Confidence": 0.90, "Calmness": 0.70, "Concentration": 0.60,
        "Doubt": 0.00, "Anxiety": 0.05, "Excitement": 0.30, 
        "Determination": 0.70
    }
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
    st.error("🔑 API Key de Hume no encontrada. Configúrala en .env o Streamlit Secrets.")
    st.stop()

# ------------------------------------------------------------
# Funciones de API de Hume
# ------------------------------------------------------------
HUME_BASE_URL = "https://api.hume.ai/v0/batch/jobs"

def start_job(api_key, file_path):
    headers = {"X-Hume-Api-Key": api_key}
    with open(file_path, "rb") as f:
        files = {"file": f}
        json_payload = json.dumps({"models": {"prosody": {}, "language": {}}})
        response = requests.post(HUME_BASE_URL, files=files, data={"json": json_payload}, headers=headers)
    if response.status_code == 200:
        return response.json()["job_id"]
    raise Exception(f"Error {response.status_code}: {response.text}")

def get_job_result(api_key, job_id):
    headers = {"X-Hume-Api-Key": api_key}
    while True:
        res = requests.get(f"{HUME_BASE_URL}/{job_id}", headers=headers).json()
        status = (res.get("state") or res.get("status") or {}).get("status", "").lower()
        if status == "completed":
            return requests.get(f"{HUME_BASE_URL}/{job_id}/predictions", headers=headers).json()
        if status in ("failed", "cancelled"):
            raise Exception("El análisis de Hume falló.")
        time.sleep(2)

# ------------------------------------------------------------
# Funciones de Análisis, Gráficos y Reportes
# ------------------------------------------------------------
def calcular_confianza_artificial(scores):
    determinacion = scores.get("Determination", 0.0)
    calma = scores.get("Calmness", 0.0)
    entusiasmo = scores.get("Excitement", 0.0)
    ansiedad = scores.get("Anxiety", 0.0)
    duda = scores.get("Doubt", 0.0)
    confianza = (determinacion * 0.4) + (calma * 0.3) + (entusiasmo * 0.2) - (ansiedad * 0.5) - (duda * 0.5)
    return max(0.0, min(1.0, confianza))

def extract_emotion_scores(predictions_payload):
    emotion_totals = {}
    segment_count = 0
    for item in predictions_payload:
        for pred in item.get("results", {}).get("predictions", []):
            for group in pred.get("models", {}).get("prosody", {}).get("grouped_predictions", []):
                for segment in group.get("predictions", []):
                    segment_count += 1
                    for emo in segment.get("emotions", []):
                        eng_name = emo.get("name", "")
                        esp_name = TRADUCCION_EMOCIONES.get(eng_name, eng_name)
                        score = emo.get("score", 0.0)
                        emotion_totals[esp_name] = emotion_totals.get(esp_name, 0.0) + score
    if segment_count == 0: return {}
    promedios = {name: total / segment_count for name, total in emotion_totals.items()}
    promedios["Confidence"] = calcular_confianza_artificial(promedios)
    return promedios

def crear_radar_plotly(radar_data, estilo):
    etiquetas = list(radar_data.keys())
    actuales = [v["actual"] * 100 for v in radar_data.values()]
    objetivos = [v["target"] * 100 for v in radar_data.values()]
    etiquetas.append(etiquetas[0])
    actuales.append(actuales[0])
    objetivos.append(objetivos[0])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=actuales, theta=etiquetas, fill='toself', name='Your Voice', line_color='#1f77b4'))
    fig.add_trace(go.Scatterpolar(r=objetivos, theta=etiquetas, fill='toself', name=f'Target: {estilo.capitalize()}', line_color='#ff7f0e'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=40, b=80), height=500, title=dict(text="Tone Comparison", x=0.5, xanchor='center')
    )
    return fig

# --- NUEVA FUNCIÓN PARA GENERAR EL REPORTE ---
def generar_reporte_html(texto, ipa_text, resultados, scores, estilo):
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; padding: 30px; line-height: 1.6; color: #333; }}
            h1 {{ color: #2C3E50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 30px; }}
            .ipa-box {{ background-color: #f8f9fa; padding: 15px; border-left: 5px solid #3498db; font-family: monospace; font-size: 16px; margin: 10px 0; }}
            .ok {{ color: #27ae60; margin-bottom: 8px; }}
            .warn {{ color: #e67e22; margin-bottom: 8px; }}
            .emotion-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; max-width: 400px; }}
            .emotion-item {{ background: #ecf0f1; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }}
            @media print {{ body {{ padding: 0; }} }}
        </style>
    </head>
    <body>
        <h1>🎤 Vocal Coach Pro - Analysis Report</h1>
        <p><strong>Target Style:</strong> {estilo.capitalize()}</p>
        
        <h2>1. Phonetic Scorecard</h2>
        <p><strong>Text:</strong> {texto}</p>
        <div class="ipa-box"><strong>IPA:</strong> {ipa_text}</div>
        
        <h2>2. Word-by-Word Stress Analysis</h2>
        <ul>
    """
    for res in resultados:
        if res['feedback']:
            if "✅" in res['feedback']:
                html += f'<li class="ok">{res["feedback"]}</li>'
            else:
                html += f'<li class="warn">{res["feedback"]}</li>'
                
    html += f"""
        </ul>
        <h2>3. Top Emotions Detected</h2>
        <div class="emotion-grid">
    """
    # Mostrar el top 4 de emociones
    for e, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:4]:
        html += f'<div class="emotion-item">{e}<br>{v*100:.1f}%</div>'
        
    html += """
        </div>
        <p style="margin-top:40px; font-size:12px; color:#7f8c8d; text-align:center;">
            Generado por English Vocal Coach Pro. (Presiona Ctrl+P o Cmd+P para guardar este reporte como PDF).
        </p>
    </body>
    </html>
    """
    return html

# ------------------------------------------------------------
# Lógica de Análisis Fonético (IPA + Librosa)
# ------------------------------------------------------------
def get_ipa_info(word):
    clean_word = word.strip(".,;:!?")
    ipa_str = ipa.convert(clean_word, keep_punct=False)
    stressed_pos = 0
    if 'ˈ' in ipa_str:
        prefix = ipa_str.split('ˈ')[0]
        vowels = "iɪeɛæɑɔoʊuʌəɚɝaɐ"
        stressed_pos = sum(1 for c in prefix if c in vowels) + 1
    return ipa_str, stressed_pos

def analyze_stress(audio_path, words_data):
    y, sr = librosa.load(audio_path, sr=22050)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    
    results = []
    for wd in words_data:
        word, start, end = wd['word'], wd['start'], wd['end']
        ipa_str, ideal_stress = get_ipa_info(word)
        
        mask = (times >= start) & (times <= end)
        if not np.any(mask): continue
        
        segment_rms = rms[mask]
        num_syllables = max(1, sum(1 for c in ipa_str if c in "iɪeɛæɑɔoʊuʌəɚɝaɐ"))
        parts = np.array_split(segment_rms, num_syllables)
        actual_stress = np.argmax([np.mean(p) for p in parts]) + 1
        
        feedback = None
        if ideal_stress > 0:
            if actual_stress == ideal_stress:
                feedback = f"✅ **{word}**: Perfect stress on syllable {actual_stress}."
            else:
                feedback = f"⚠️ **{word}**: Stressed syllable {actual_stress}, but IPA suggests {ideal_stress} ({ipa_str})."
        
        results.append({"word": word, "feedback": feedback})
    return results

# ------------------------------------------------------------
# UI Streamlit
# ------------------------------------------------------------
st.sidebar.header("Settings")
estilo = st.sidebar.selectbox("Target Style", ("persuasive", "direct", "expert"))
archivo_subido = st.file_uploader("Upload English Audio (WAV/MP3)", type=["wav", "mp3"])

if archivo_subido:
    _, extension = os.path.splitext(archivo_subido.name)
    if not extension: extension = ".wav"
    
    audio_bytes = archivo_subido.read()
    st.audio(audio_bytes, format=f"audio/{extension.replace('.', '')}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    if st.button("Start Pro Analysis"):
        with st.spinner("Analyzing audio and phonetics..."):
            try:
                job_id = start_job(api_key, audio_path)
                predictions = get_job_result(api_key, job_id)
                scores = extract_emotion_scores(predictions)
                
                words_data = []
                for item in predictions:
                    for pred in item.get("results", {}).get("predictions", []):
                        models = pred.get("models", {})
                        for model_name in ["prosody", "language"]:
                            encontro_texto = False
                            for group in models.get(model_name, {}).get("grouped_predictions", []):
                                for seg in group.get("predictions", []):
                                    if "words" in seg and isinstance(seg["words"], list) and len(seg["words"]) > 0:
                                        for w in seg["words"]:
                                            inicio = w.get('begin', w.get('start', 0))
                                            fin = w.get('end', 0)
                                            words_data.append({'word': w.get('word',''), 'start': inicio, 'end': fin})
                                        encontro_texto = True
                                    elif "text" in seg and seg["text"].strip():
                                        text = seg["text"].strip()
                                        tiempo = seg.get("time", {})
                                        inicio = tiempo.get("begin", tiempo.get("start", 0))
                                        fin = tiempo.get("end", 0)
                                        words = text.split()
                                        if words:
                                            duracion_por_palabra = (fin - inicio) / len(words)
                                            for i, w in enumerate(words):
                                                words_data.append({'word': w, 'start': inicio + (i * duracion_por_palabra), 'end': inicio + ((i + 1) * duracion_por_palabra)})
                                        encontro_texto = True
                            if encontro_texto: break

                if not words_data:
                    st.warning("⚠️ Hume AI processed the audio, but detected no speech. Is it too noisy or silent?")
                else:
                    # Resultados de Transcripción e IPA
                    st.subheader("🎼 Phonetic Scorecard (IPA)")
                    full_text = " ".join([w['word'] for w in words_data])
                    ipa_text = ipa.convert(full_text)
                    st.markdown(f"**Text:** {full_text}")
                    st.info(f"**IPA:** {ipa_text}")

                    # Resultados de Acento
                    st.markdown("---")
                    st.subheader("🔍 Word-by-Word Stress Analysis")
                    results = analyze_stress(audio_path, words_data)
                    for res in results:
                        if res['feedback']:
                            if "✅" in res['feedback']: st.success(res['feedback'])
                            else: st.warning(res['feedback'])

                # Kanban de Emociones
                if scores:
                    st.markdown("---")
                    st.subheader("📋 Emotional Intensity Board")
                    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("### 🔴 High")
                        for e, v in sorted_emotions:
                            if v >= 0.25:
                                with st.container(border=True): st.markdown(f"**{e}**\n### {v*100:.1f}%")
                    with c2:
                        st.markdown("### 🟡 Medium")
                        for e, v in sorted_emotions:
                            if 0.10 <= v < 0.25:
                                with st.container(border=True): st.markdown(f"**{e}**\n### {v*100:.1f}%")
                    with c3:
                        st.markdown("### ⚪ Low")
                        for e, v in sorted_emotions[:15]:
                            if v < 0.10:
                                with st.container(border=True): st.markdown(f"**{e}**\n### {v*100:.1f}%")

                    # Radar Chart
                    st.markdown("---")
                    st.subheader("📈 Voice Profile Map")
                    radar_data = {e: {"actual": scores.get(e, 0.0), "target": v} for e, v in IDEAL_PROFILES[estilo].items()}
                    fig = crear_radar_plotly(radar_data, estilo)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

                    # --- BOTÓN DE EXPORTACIÓN ---
                    st.markdown("---")
                    st.subheader("📥 Download Analysis")
                    st.info("💡 **Tip:** Descarga el reporte HTML, ábrelo en tu navegador web y presiona **Ctrl + P** (o Comando + P) para guardarlo perfectamente formateado como PDF.")
                    
                    html_report = generar_reporte_html(full_text, ipa_text, results, scores, estilo)
                    
                    st.download_button(
                        label="📄 Descargar Reporte Completo (HTML para PDF)",
                        data=html_report,
                        file_name="VocalCoach_Report.html",
                        mime="text/html"
                    )

            except Exception as e:
                st.error(f"Error during analysis: {e}")
            finally:
                if os.path.exists(audio_path): 
                    os.unlink(audio_path)
                if 'audio_bytes' in locals():
                    del audio_bytes
                gc.collect()
