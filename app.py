# app.py - Entrenador Vocal con Hume AI + Partitura IPA + Comparación sílaba tónica (librosa)
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
import soundfile as sf
from collections import defaultdict

# ------------------------------------------------------------
# Configuración de la página
# ------------------------------------------------------------
st.set_page_config(page_title="Entrenador Vocal Pro", page_icon="🎤", layout="wide")
st.title("🎤 Entrenador Vocal Pro (con análisis de acento)")
st.markdown("Analiza tu voz para sonar más **persuasiva**, **directa** o **experta**.\n"
            "Ahora con **feedback palabra por palabra** comparando tu énfasis con el IPA ideal.")

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
# Funciones API Hume
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

# ------------------------------------------------------------
# Extracción de emociones (igual que antes)
# ------------------------------------------------------------
def calcular_confianza_artificial(scores):
    determinacion = scores.get("Determinación", 0.0)
    calma = scores.get("Calma", 0.0)
    entusiasmo = scores.get("Entusiasmo", 0.0)
    ansiedad = scores.get("Ansiedad", 0.0)
    duda = scores.get("Duda", 0.0)
    confianza = (determinacion * 0.4) + (calma * 0.3) + (entusiasmo * 0.2) - (ansiedad * 0.5) - (duda * 0.5)
    return max(0.0, min(1.0, confianza))

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
    promedios["Confianza"] = calcular_confianza_artificial(promedios)
    return promedios

# ------------------------------------------------------------
# NUEVA FUNCIÓN: Extraer palabras con tiempos desde Hume
# ------------------------------------------------------------
def extract_words_with_timing(predictions_payload):
    """
    Devuelve una lista de diccionarios con:
      - 'word': texto de la palabra
      - 'start': tiempo de inicio en segundos
      - 'end': tiempo de fin en segundos
    """
    words = []
    for item in predictions_payload:
        if "results" in item and "predictions" in item["results"]:
            for pred in item["results"]["predictions"]:
                if "models" in pred and "language" in pred["models"]:
                    for group in pred["models"]["language"].get("grouped_predictions", []):
                        for segment in group.get("predictions", []):
                            for word_obj in segment.get("words", []):
                                word = word_obj.get("word", "").strip()
                                start = word_obj.get("start")
                                end = word_obj.get("end")
                                if word and start is not None and end is not None:
                                    words.append({"word": word, "start": start, "end": end})
    return words

# ------------------------------------------------------------
# NUEVA FUNCIÓN: Partitura IPA (para frase completa y por palabra)
# ------------------------------------------------------------
def generar_partitura_ipa(texto):
    if not texto:
        return ""
    try:
        return ipa.convert(texto, keep_punct=False, retrieve_all=False)
    except Exception:
        return texto

def get_ipa_word(palabra):
    """Devuelve la transcripción IPA de una palabra (limpia de puntuación)."""
    limpia = palabra.strip(".,;:!?¡¿()[]{}\"'")
    if not limpia:
        return ""
    try:
        return ipa.convert(limpia, keep_punct=False, retrieve_all=False)
    except Exception:
        return limpia

# ------------------------------------------------------------
# NUEVA FUNCIÓN: Detectar sílaba tónica en una cadena IPA
# ------------------------------------------------------------
def find_stressed_syllable_index(ipa_str):
    """
    Busca la primera sílaba con acento primario (ˈ) o secundario (ˌ).
    Devuelve el índice dentro de la cadena donde comienza esa sílaba.
    Si no hay, devuelve -1.
    """
    # Buscar ˈ o ˌ
    for i, ch in enumerate(ipa_str):
        if ch in ('ˈ', 'ˌ'):
            # La sílaba tónica comienza después del símbolo
            return i + 1
    return -1

def estimate_syllable_count(word, sr=22050, hop_length=512):
    """Estimación grosera del número de sílabas basada en vocales del IPA.
       Fallback: número de vocales en la palabra original.
    """
    ipa_word = get_ipa_word(word)
    # Contar vocales IPA básicas (simplificado)
    vowels_ipa = set("iɪeɛæɑɔoʊuʌəɚɝaɐ")
    count = sum(1 for c in ipa_word if c in vowels_ipa)
    if count == 0:
        # Contar vocales en inglés (a,e,i,o,u) como fallback
        count = sum(1 for c in word.lower() if c in 'aeiou')
    return max(1, count)

# ------------------------------------------------------------
# NUEVA FUNCIÓN PRINCIPAL: Comparar acento prosódico con audio
# ------------------------------------------------------------
def analizar_acento_palabras(audio_path, words_data):
    """
    words_data: lista de dicts con word, start, end.
    audio_path: ruta al archivo de audio.
    Devuelve una lista de dicts con:
      - 'word': palabra original
      - 'ipa': IPA ideal
      - 'stressed_syllable': número de sílaba tónica (1-indexed) o 0 si no hay
      - 'syllable_count': número total de sílabas estimadas
      - 'feedback': texto de sugerencia si la energía máxima no coincide con la tónica, o None
      - 'energy_peak_position': en qué intervalo (sílaba) ocurrió el pico de energía
    """
    # Cargar audio
    y, sr = librosa.load(audio_path, sr=22050)
    # Calcular RMS (energía) frame por frame
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    frames = np.arange(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    resultados = []

    for wd in words_data:
        word = wd["word"]
        start = wd["start"]
        end = wd["end"]

        # 1. IPA y sílaba tónica
        ipa_word = get_ipa_word(word)
        if not ipa_word:
            resultados.append({"word": word, "ipa": "", "stressed_syllable": 0,
                               "syllable_count": 1, "feedback": None,
                               "energy_peak_position": 0})
            continue

        tonic_idx = find_stressed_syllable_index(ipa_word)
        # Determinar número de sílaba tónica (1-indexed)
        # Necesitamos dividir IPA en sílabas. Aproximación simple: las sílabas suelen estar separadas por '.' en IPA.
        # Pero eng_to_ipa no siempre pone '.'; podemos insertarlos con una función.
        # Por simplicidad, contaremos las vocales para estimar cuántas sílabas hay.
        num_silabas = estimate_syllable_count(word)

        # Determinar en qué posición está la sílaba tónica (primera, segunda, etc.)
        # Si encontramos el símbolo de acento, vemos cuántas vocales hay antes de él (incluyendo las de la sílaba tónica).
        # Más sencillo: si el acento está en la primera sílaba, la vocal inmediatamente después es la primera sílaba tónica.
        if tonic_idx == -1:
            stressed_pos = 0  # No hay acento marcado
        else:
            # Contar cuántas vocales hay antes de tonic_idx (aproximación)
            prefix = ipa_word[:tonic_idx]
            # Contamos símbolos vocálicos en prefix
            vocals = sum(1 for c in prefix if c in "iɪeɛæɑɔoʊuʌəɚɝaɐ")
            # La sílaba tónica será vocals+1 (porque las vocales anteriores cuentan como sílabas completas)
            stressed_pos = vocals + 1
        # Asegurar que stressed_pos no exceda num_silabas
        if stressed_pos > num_silabas:
            stressed_pos = num_silabas

        # 2. Extraer segmento de energía para esta palabra
        # Encontrar índices de frames que caen dentro de [start, end]
        mask = (times >= start) & (times <= end)
        if not np.any(mask):
            resultados.append({"word": word, "ipa": ipa_word,
                               "stressed_syllable": stressed_pos if tonic_idx != -1 else 0,
                               "syllable_count": num_silabas,
                               "feedback": None, "energy_peak_position": 0})
            continue

        segment_rms = rms[mask]
        segment_times = times[mask]

        if len(segment_rms) == 0:
            resultados.append(...)
            continue

        # 3. Dividir el segmento en intervalos de sílabas (por onsets o por división uniforme)
        # Usaremos onset detection dentro del segmento para dividir.
        # Si no se detectan onsets, dividimos uniformemente.
        try:
            onset_frames = librosa.onset.onset_detect(
                y=y[int(start * sr):int(end * sr)],
                sr=sr,
                hop_length=hop_length,
                backtrack=True,
                units='frames'
            )
            # Convertir onset frames a tiempos absolutos
            onset_times_abs = [start + librosa.frames_to_time(f, sr=sr, hop_length=hop_length)
                               for f in onset_frames]
        except Exception:
            onset_times_abs = []

        # Añadir el inicio y final de la palabra
        boundaries = [start] + onset_times_abs + [end]
        # Filtrar boundaries ordenadas y únicas
        boundaries = sorted(list(set(boundaries)))

        # Si hay menos de 3 boundaries (solo start y end), dividimos uniformemente en num_silabas
        if len(boundaries) <= 2:
            # División uniforme
            step = (end - start) / num_silabas
            boundaries = [start + i * step for i in range(num_silabas + 1)]

        # 4. Calcular energía media en cada intervalo silábico
        energy_per_syllable = []
        for i in range(len(boundaries) - 1):
            t0 = boundaries[i]
            t1 = boundaries[i + 1]
            mask_int = (segment_times >= t0) & (segment_times < t1)
            if np.any(mask_int):
                energy_per_syllable.append(np.mean(segment_rms[mask_int]))
            else:
                energy_per_syllable.append(0.0)

        # 5. Encontrar el intervalo con mayor energía
        if energy_per_syllable:
            peak_syl_idx = np.argmax(energy_per_syllable)  # 0-indexed
            peak_syl_number = peak_syl_idx + 1  # 1-indexed
        else:
            peak_syl_number = 0

        # 6. Comparar con la sílaba tónica esperada
        feedback = None
        if stressed_pos > 0 and peak_syl_number > 0 and peak_syl_number != stressed_pos:
            # Sugerencia
            feedback = (f"⚠️ En '{word}', el acento debería estar en la sílaba {stressed_pos}"
                        f" (según IPA '{ipa_word}'), pero tu energía máxima está en la sílaba {peak_syl_number}."
                        f" Intenta enfatizar la sílaba correcta.")
        elif stressed_pos > 0 and peak_syl_number == stressed_pos:
            feedback = f"✅ '{word}' bien enfatizada."

        resultados.append({
            "word": word,
            "ipa": ipa_word,
            "stressed_syllable": stressed_pos if tonic_idx != -1 else 0,
            "syllable_count": num_silabas,
            "feedback": feedback,
            "energy_peak_position": peak_syl_number
        })

    return resultados

# ------------------------------------------------------------
# Visualización clásica de feedback global
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
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
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

                    # --- Transcripción con tiempos ---
                    words_data = extract_words_with_timing(predictions)
                    textos_transcritos = [w["word"] for w in words_data]
                    texto_completo = " ".join(textos_transcritos)
                    if texto_completo:
                        st.info(f"📝 **Texto completo transcrito:** \"{texto_completo}\"")
                    else:
                        st.warning("⚠️ No se detectaron palabras claras para transcribir.")

                    # --- Partitura IPA global ---
                    if texto_completo:
                        st.markdown("---")
                        st.subheader("🎼 Partitura Fonética (IPA) ideal")
                        ipa_transcrito = generar_partitura_ipa(texto_completo)
                        st.markdown(f"""
                        <div style="background-color:#F0F2F6; padding:20px; border-radius:10px; font-family:'Courier New', monospace; font-size:24px; line-height:1.8; letter-spacing:2px; word-spacing:12px;">
                        {ipa_transcrito}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("Los símbolos ˈ indican sílaba tónica, ˌ acento secundario, . separa sílabas.")

                    # --- NUEVA SECCIÓN: Comparación de acento con librosa ---
                    if words_data:
                        st.markdown("---")
                        st.subheader("🔍 Análisis de acento por palabra (tu voz vs. IPA)")
                        with st.spinner("Calculando energía y comparando sílabas..."):
                            try:
                                analisis = analizar_acento_palabras(audio_path, words_data)
                            except Exception as e:
                                st.warning(f"No se pudo realizar el análisis de acento: {e}")
                                analisis = []

                        if analisis:
                            for a in analisis:
                                if a.get("feedback"):
                                    if a["feedback"].startswith("✅"):
                                        st.success(a["feedback"])
                                    else:
                                        st.warning(a["feedback"])
                                else:
                                    if a.get("stressed_syllable") == 0:
                                        st.caption(f"ℹ️ '{a['word']}' no tiene acento marcado en IPA.")
                                    else:
                                        st.info(f"ℹ️ '{a['word']}' – sin conclusión (datos insuficientes).")
                        else:
                            st.info("No se encontraron palabras para analizar.")

                    # --- Kanban de emociones ---
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

                    # --- Radar ---
                    st.markdown("---")
                    st.subheader("📈 Mapa de Perfil Vocal")
                    radar_data = {e: {"actual": scores.get(e, 0.0), "target": v} for e, v in IDEAL_PROFILES[estilo].items()}
                    fig = crear_radar_plotly(radar_data, estilo)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

                    # --- Tips globales ---
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
