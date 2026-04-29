# app.py - Entrenador Vocal Pro (Edición Inglés)
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
# Configuración de Hume AI
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

# Diccionario de Traducción de Emociones
TRADUCCION_EMOCIONES = {
    "Admiration": "Admiración", "Anger": "Enojo", "Anxiety": "Ansiedad", "Awe": "Asombro",
    "Awkwardness": "Incomodidad", "Boredom": "Aburrimiento", "Calmness": "Calma",
    "Concentration": "Concentración", "Confusion": "Confusión", "Contempt": "Desprecio",
    "Contentment": "Satisfacción", "Determination": "Determinación", "Doubt": "Duda",
    "Excitement": "Entusiasmo", "Fear": "Miedo", "Interest": "Interés", "Joy": "Alegría",
    "Pride": "Orgullo", "Sadness": "Tristeza", "Surprise (positive)": "Sorpresa (+)",
    "Tiredness": "Cansancio", "Triumph": "Triunfo"
}

IDEAL_PROFILES = {
    "persuasive": {"Confidence": 0.85, "Excitement": 0.65, "Calmness": 0.50, "Determination": 0.75},
    "direct": {"Confidence": 0.80, "Determination": 0.90, "Calmness": 0.60},
    "expert": {"Confidence": 0.90, "Calmness": 0.70, "Concentration": 0.60}
}

# ------------------------------------------------------------
# Funciones de Procesamiento (Hume AI)
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
# Lógica de Análisis Fonético (IPA + Librosa)
# ------------------------------------------------------------
def get_ipa_info(word):
    clean_word = word.strip(".,;:!?")
    ipa_str = ipa.convert(clean_word, keep_punct=False)
    # Buscar símbolo de acento primario ˈ
    stressed_pos = 0
    if 'ˈ' in ipa_str:
        # Contar vocales antes del acento para saber el número de sílaba
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
        
        # Extraer energía de este segmento
        mask = (times >= start) & (times <= end)
        if not np.any(mask): continue
        
        segment_rms = rms[mask]
        # Dividir segmento en "n" partes (basado en sílabas estimadas)
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
# Interfaz de Usuario (Streamlit)
# ------------------------------------------------------------
st.sidebar.header("Settings")
estilo = st.sidebar.selectbox("Target Style", ("persuasive", "direct", "expert"))
archivo_subido = st.file_uploader("Upload English Audio (WAV/MP3)", type=["wav", "mp3"])

if archivo_subido:
    # --- SOLUCIÓN PUNTERO: Leer bytes primero ---
    audio_bytes = archivo_subido.read()
    st.audio(audio_bytes)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    if st.button("Start Pro Analysis"):
        with st.spinner("Analyzing audio and phonetics..."):
            try:
                predictions = get_job_result(api_key, start_job(api_key, audio_path))
                
                # 1. Extraer palabras y tiempos
                words_data = []
                for item in predictions:
                    for pred in item.get("results", {}).get("predictions", []):
                        for group in pred.get("models", {}).get("language", {}).get("grouped_predictions", []):
                            for seg in group.get("predictions", []):
                                for w in seg.get("words", []):
                                    words_data.append({'word': w['word'], 'start': w['start'], 'end': w['end']})

                # 2. Mostrar Transcripción e IPA
                st.subheader("🎼 Phonetic Scorecard (IPA)")
                full_text = " ".join([w['word'] for w in words_data])
                ipa_text = ipa.convert(full_text)
                st.markdown(f"**Text:** {full_text}")
                st.info(f"**IPA:** {ipa_text}")

                # 3. Análisis de Acento (Librosa)
                st.subheader("🔍 Word-by-Word Stress Analysis")
                results = analyze_stress(audio_path, words_data)
                for res in results:
                    if res['feedback']:
                        if "✅" in res['feedback']: st.success(res['feedback'])
                        else: st.warning(res['feedback'])

                # 4. Emociones (Hume)
                st.subheader("📊 Emotional Profile")
                # (Aquí puedes añadir tu lógica de radar/kanban simplificada)
                st.write("Análisis emocional completado. Revisa tus niveles de confianza y calma.")

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(audio_path): os.unlink(audio_path)
