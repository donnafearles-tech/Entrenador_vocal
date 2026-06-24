# app.py - English Vocal Coach Pro (Hume EVI WebSockets)
import streamlit as st
import os
import json
import tempfile
import base64
import asyncio
import websockets
import plotly.graph_objects as go
import eng_to_ipa as ipa
import librosa
import numpy as np
import gc

# ------------------------------------------------------------
# Configuración de la página
# ------------------------------------------------------------
st.set_page_config(page_title="English Vocal Coach Pro", page_icon="🎤", layout="wide")
st.title("🎤 English Vocal Coach Pro (Live EVI)")
st.markdown("""
Analiza tu pronunciación y tono en inglés conectándote en vivo con **Hume EVI**. 
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
    "Tiredness": "Tiredness", "Triumph": "Triumph", "Confidence": "Confidence"
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
    st.error("🔑 API Key de Hume no encontrada. Configúrala en .env.")
    st.stop()

# ------------------------------------------------------------
# Funciones Asíncronas para Hume EVI (WebSockets)
# ------------------------------------------------------------
def calcular_confianza_artificial(scores):
    determinacion = scores.get("Determination", 0.0)
    calma = scores.get("Calmness", 0.0)
    entusiasmo = scores.get("Excitement", 0.0)
    ansiedad = scores.get("Anxiety", 0.0)
    duda = scores.get("Doubt", 0.0)
    confianza = (determinacion * 0.4) + (calma * 0.3) + (entusiasmo * 0.2) - (ansiedad * 0.5) - (duda * 0.5)
    return max(0.0, min(1.0, confianza))

async def analyze_audio_with_evi(api_key, audio_path):
    uri = f"wss://api.hume.ai/v0/evi/chat?api_key={api_key}"
    
    transcribed_text = ""
    emotion_scores_raw = {}
    
    try:
        async with websockets.connect(uri) as websocket:
            # 1. Leer y codificar el archivo de audio
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            
            # 2. Enviar el paquete de audio a EVI
            audio_msg = {
                "type": "audio_input",
                "data": base64_audio
            }
            await websocket.send(json.dumps(audio_msg))
            
            # 3. Escuchar las respuestas del WSS
            while True:
                try:
                    # 🚀 CAMBIO 1: Aumentamos el timeout a 30 segundos
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    data = json.loads(response)
                    
                    # 🚀 CAMBIO 2: Imprimir en la consola el tipo de evento para depurar
                    print(f"[HUME EVI EVENT] Recibido evento de tipo: {data.get('type')}")
                    
                    if data.get("type") == "user_message":
                        message = data.get("message", {})
                        transcribed_text = message.get("content", "")
                        
                        models = data.get("models", {})
                        prosody = models.get("prosody", {})
                        scores = prosody.get("scores", {})
                        
                        if scores:
                            emotion_scores_raw = scores
                            break # Tenemos lo que necesitamos, salimos
                            
                    elif data.get("type") == "error":
                        # Si Hume tira un error interno, lo mostramos
                        raise Exception(data.get("message", "Error desconocido en EVI"))
                        
                except asyncio.TimeoutError:
                    print("[HUME EVI TIMEOUT] Se agotaron los 30 segundos esperando a EVI.")
                    break 
                    
    except Exception as e:
        raise Exception(f"Fallo en la conexión WebSocket: {str(e)}")
        
    return transcribed_text, emotion_scores_raw

# ------------------------------------------------------------
# Análisis Fonético y Gráficos
# ------------------------------------------------------------
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

def get_ipa_info(word):
    clean_word = word.strip(".,;:!?")
    ipa_str = ipa.convert(clean_word, keep_punct=False)
    stressed_pos = 0
    if 'ˈ' in ipa_str:
        prefix = ipa_str.split('ˈ')[0]
        vowels = "iɪeɛæɑɔoʊuʌəɚɝaɐ"
        stressed_pos = sum(1 for c in prefix if c in vowels) + 1
    return ipa_str, stressed_pos

def analyze_stress(audio_path, full_text):
    # Ya no tenemos timestamps exactos por palabra desde EVI de forma nativa sin procesamiento pesado,
    # así que aproximamos el análisis dividiendo el audio total
    y, sr = librosa.load(audio_path, sr=22050)
    rms = librosa.feature.rms(y=y)[0]
    
    words = full_text.split()
    if not words: return []
    
    chunk_size = len(rms) // len(words)
    results = []
    
    for i, word in enumerate(words):
        ipa_str, ideal_stress = get_ipa_info(word)
        segment_rms = rms[i*chunk_size : (i+1)*chunk_size]
        
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

def generar_consejos_enfasis(scores, estilo):
    consejos = []
    ideal = IDEAL_PROFILES[estilo]
    
    if scores.get("Confidence", 0) < ideal.get("Confidence", 0.8) - 0.1:
        consejos.append("📉 **Confianza baja**: evita el 'uptalk' (tono ascendente al final). Termina las frases con tono descendente.")
    if scores.get("Determination", 0) < ideal.get("Determination", 0.7) - 0.15:
        consejos.append("💪 **Determinación insuficiente**: alarga ligeramente las vocales en palabras clave y sube el volumen.")
    if scores.get("Excitement", 0) < ideal.get("Excitement", 0.5) - 0.15:
        consejos.append("⚡ **Voz plana**: varía el pitch. Sube el tono en las palabras importantes.")
    if scores.get("Anxiety", 0) > ideal.get("Anxiety", 0.1) + 0.1:
        consejos.append("😟 **Nerviosismo detectado**: respira profundamente antes de empezar.")
        
    if not consejos:
        consejos.append("✅ ¡Buen trabajo! El análisis en tiempo real muestra que estás alineada con el perfil ideal.")
    return consejos

# ------------------------------------------------------------
# UI Streamlit
# ------------------------------------------------------------
st.sidebar.header("Settings")
estilo = st.sidebar.selectbox("Target Style", ("persuasive", "direct", "expert"))

# --- NUEVO: Selección del método de entrada ---
metodo_entrada = st.sidebar.radio("Input Method", ["Microphone (Live)", "Upload File"])

archivo_audio = None

if metodo_entrada == "Microphone (Live)":
    # Componente nativo de Streamlit para grabar desde el navegador
    archivo_audio = st.audio_input("🎤 Record your voice in English")
else:
    archivo_audio = st.file_uploader("Upload English Audio (WAV/MP3)", type=["wav", "mp3"])

# --- Lógica principal de procesamiento ---
if archivo_audio:
    # Obtener extensión de forma segura dependiendo del método de entrada
    nombre_archivo = getattr(archivo_audio, 'name', 'grabacion.wav')
    _, extension = os.path.splitext(nombre_archivo)
    if not extension: extension = ".wav"
    
    audio_bytes = archivo_audio.read()
    
    # st.audio_input ya muestra un reproductor, solo lo mostramos extra si es archivo subido
    if metodo_entrada == "Upload File":
        st.audio(audio_bytes, format=f"audio/{extension.replace('.', '')}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    if st.button("Start Live EVI Analysis"):
        with st.spinner("Conectando con Hume EVI a través de WebSockets..."):
            try:
                # 🚀 Ejecutar la función asíncrona dentro del bucle de eventos sincrónico de Streamlit
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                # Tu función actual que hace el llamado a wss://api.hume.ai/v0/evi/chat
                transcripcion, raw_scores = loop.run_until_complete(analyze_audio_with_evi(api_key, audio_path))
                
                if not raw_scores:
                    st.error("⚠️ El socket se cerró sin devolver métricas de prosodia. Verifica el audio.")
                else:
                    st.success("✅ Análisis EVI completado")
                    scores = process_evi_scores(raw_scores)
                    
                    # Consejos
                    st.markdown("---")
                    st.subheader("🎯 Live Emphasis Tips")
                    consejos = generar_consejos_enfasis(scores, estilo)
                    for tip in consejos:
                        st.info(tip)
                    
                    # IPA
                    st.markdown("---")
                    st.subheader("🎼 Phonetic Scorecard (IPA)")
                    ipa_text = ipa.convert(transcripcion)
                    st.markdown(f"**Text:** {transcripcion}")
                    st.info(f"**IPA:** {ipa_text}")

                    # Acento y Librosa
                    st.markdown("---")
                    st.subheader("🔍 Estimated Stress Analysis")
                    results = analyze_stress(audio_path, transcripcion)
                    for res in results:
                        if res['feedback']:
                            if "✅" in res['feedback']: st.success(res['feedback'])
                            else: st.warning(res['feedback'])

                    # Kanban
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

            except Exception as e:
                st.error(f"Error during EVI WSS analysis: {e}")
            finally:
                if os.path.exists(audio_path): 
                    os.unlink(audio_path)
                if 'audio_bytes' in locals():
                    del audio_bytes
                import gc
                gc.collect()
