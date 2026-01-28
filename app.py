import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import random
import glob
import cv2
import av
from PIL import Image

# Biblioteki do streamingu wideo w chmurze
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Importy AI
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from deepface import DeepFace

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="System Analizy Emocji (Hybrydowy)", layout="wide")

# ==========================================
# KONFIGURACJA 1: TWÓJ MODEL (BADANIA)
# ==========================================
MODEL_PATH = 'moj_model_fer.h5'
DATASET_TEST_PATH = './dane_fer/test'
DEFAULT_USER_FOLDER = 'test_real'

CLASSES = ['angry', 'happy', 'sad']
TRANSLATION = {'angry': 'ZŁOŚĆ', 'happy': 'RADOŚĆ', 'sad': 'SMUTEK'}
COLORS = {'angry': '#FF4B4B', 'happy': '#2ECC71', 'sad': '#3498DB'}

# ==========================================
# KONFIGURACJA 2: KAMERA (DEEPFACE)
# ==========================================
# Emocje, które nas interesują
TARGET_EMOTIONS = {'angry': 'ZŁOŚĆ', 'happy': 'RADOŚĆ', 'sad': 'SMUTEK'}

# Kolory BGR dla OpenCV (Ramki wideo)
BOX_COLORS = {
    'ZŁOŚĆ': (0, 0, 255),  # Czerwony
    'RADOŚĆ': (0, 255, 0),  # Zielony
    'SMUTEK': (255, 0, 0),  # Niebieski
    'NIEOKREŚLONY': (200, 200, 200)
}


# --- FUNKCJE POMOCNICZE (TWÓJ MODEL) ---
@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH): return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path):
    try:
        img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
        img_array = img_to_array(img)
        img_array /= 255.0
        return np.expand_dims(img_array, axis=0)
    except:
        return None


def get_dataset_images(limit=None):
    search_path = os.path.join(DATASET_TEST_PATH, '**', '*.jpg')
    found_files = glob.glob(search_path, recursive=True)
    if not found_files: return []
    if limit and len(found_files) > limit: return random.sample(found_files, limit)
    return found_files


# ==========================================
# KLASA DO PRZETWARZANIA WIDEO (WEBRTC)
# ==========================================
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_label = "Szukam..."
        self.last_color = (255, 255, 255)
        # Ładujemy detektor twarzy raz przy starcie
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        # Konwersja klatki na format OpenCV (numpy)
        img = frame.to_ndarray(format="bgr24")

        # 1. Wykrywanie twarzy (Szybkie)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Rysujemy ramkę
            cv2.rectangle(img, (x, y), (x + w, y + h), self.last_color, 2)

            # 2. Analiza DeepFace co 10 klatek (dla płynności)
            if self.frame_count % 10 == 0:
                try:
                    # Wycinamy twarz
                    face_roi = img[y:y + h, x:x + w]

                    # DeepFace
                    res = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    if isinstance(res, list): res = res[0]

                    all_emotions = res['emotion']  # np. {'angry': 20, 'happy': 5, 'neutral': 70...}

                    # --- FILTROWANIE (TYLKO 3 EMOCJE) ---
                    # Wyciągamy punkty tylko dla angry, happy, sad
                    scores = {k: all_emotions.get(k, 0) for k in TARGET_EMOTIONS.keys()}

                    total = sum(scores.values())
                    if total > 0:
                        # Znajdź tę, która ma najwięcej punktów wśród naszej trójki
                        winner_key = max(scores, key=scores.get)
                        winner_conf = scores[winner_key] / total  # Normalizacja

                        self.last_label = f"{TARGET_EMOTIONS[winner_key]} ({winner_conf:.0%})"
                        self.last_color = BOX_COLORS.get(TARGET_EMOTIONS[winner_key], (200, 200, 200))
                    else:
                        self.last_label = "Inna emocja"
                        self.last_color = (200, 200, 200)

                except Exception:
                    pass

            # Podpis nad głową (zawsze, nawet jak nie analizujemy w tej klatce)
            cv2.putText(img, self.last_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.last_color, 2)

        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================================================
# GŁÓWNY INTERFEJS
# =========================================================

st.sidebar.title("🎛️ Panel Sterowania")
app_mode = st.sidebar.selectbox(
    "Wybierz tryb:",
    ["📂 Badanie (Mój Model - Pliki)", "📹 Kamera (DeepFace - Live/Foto)"]
)

# ---------------------------------------------------------
# TRYB 1: BADANIE (TWÓJ MODEL) - BEZ ZMIAN
# ---------------------------------------------------------
if app_mode == "📂 Badanie (Mój Model - Pliki)":
    st.title("🧠 Badanie: Twój Model .h5")
    model = load_ai_model()
    if not model:
        st.error(f"Brak modelu {MODEL_PATH}")
        st.stop()

    source = st.sidebar.radio("Źródło:", ("📂 Folder test_real", "📚 Dataset"))
    img_paths = []

    if source == "📂 Folder test_real":
        folder = st.sidebar.text_input("Folder:", DEFAULT_USER_FOLDER)
        if os.path.exists(folder):
            img_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('jpg', 'png'))]
    else:
        if os.path.exists(DATASET_TEST_PATH):
            limit = st.sidebar.number_input("Ile zdjęć?", 10, 500, 50)
            img_paths = get_dataset_images(limit)

    if img_paths:
        if st.button("Uruchom analizę"):
            results = []
            bar = st.progress(0)
            for i, p in enumerate(img_paths):
                proc = preprocess_image(p)
                if proc is not None:
                    pred = model.predict(proc, verbose=0)[0]
                    idx = np.argmax(pred)
                    label = CLASSES[idx]

                    # Ground Truth
                    folder_name = os.path.basename(os.path.dirname(p))
                    true_lbl = TRANSLATION.get(folder_name, "-") if folder_name in CLASSES else "-"

                    results.append({
                        "Plik": os.path.basename(p),
                        "Wykryta": TRANSLATION[label],
                        "Pewnosc": np.max(pred),
                        "Prawda": true_lbl,
                        "Poprawne": (true_lbl == TRANSLATION[label]) if true_lbl != "-" else False,
                        "Ścieżka": p,
                        "Raw_Angry": pred[0], "Raw_Happy": pred[1], "Raw_Sad": pred[2]
                    })
                bar.progress((i + 1) / len(img_paths))

            st.session_state['df_res'] = pd.DataFrame(results)

    if 'df_res' in st.session_state:
        df = st.session_state['df_res']
        t1, t2, t3 = st.tabs(["Raport", "Przeglądarka", "Bayes"])

        with t1:
            st.write(f"Zanalizowano {len(df)} plików.")
            fig = px.pie(df, names='Wykryta', color='Wykryta',
                         color_discrete_map={v: COLORS[k] for k, v in TRANSLATION.items()})
            st.plotly_chart(fig)
            st.dataframe(df)

        with t2:
            st.write("Galeria (Posortowana: Poprawne -> Błędne)")
            df_s = df.sort_values(by=['Poprawne', 'Pewnosc'], ascending=[False, False])
            cols = st.columns(5)
            for i, row in df_s.head(20).iterrows():
                cols[i % 5].image(row['Ścieżka'], caption=f"{row['Wykryta']} ({row['Pewnosc']:.0%})")

        with t3:
            st.write("Symulacja Bayesa")
            sel = st.selectbox("Plik:", df['Plik'])
            row = df[df['Plik'] == sel].iloc[0]
            st.image(row['Ścieżka'], width=150)

            pa = st.slider("Szansa Złość", 0.0, 1.0, 0.33)
            ph = st.slider("Szansa Radość", 0.0, 1.0, 0.33)
            ps = st.slider("Szansa Smutek", 0.0, 1.0, 0.33)
            priors = np.array([pa, ph, ps])
            priors /= (priors.sum() + 1e-9)

            like = np.array([row['Raw_Angry'], row['Raw_Happy'], row['Raw_Sad']])
            post = like * priors
            post /= post.sum()

            fig = go.Figure(data=[
                go.Bar(name='Model', x=list(TRANSLATION.values()), y=like),
                go.Bar(name='Bayes', x=list(TRANSLATION.values()), y=post)
            ])
            st.plotly_chart(fig)

# ---------------------------------------------------------
# TRYB 2: KAMERA (LIVE + FOTO) - WEBRTC & DEEPFACE
# ---------------------------------------------------------
elif app_mode == "📹 Kamera (DeepFace - Live/Foto)":
    st.title("📹 Detekcja Live (WebRTC)")
    st.markdown("Wybierz metodę. **Live Stream** może chwilę ładować się na starcie.")

    method = st.radio("Metoda:", ["🔴 Live Stream (Wideo)", "📸 Pojedyncze Zdjęcie"])

    if method == "🔴 Live Stream (Wideo)":
        st.write("Kliknij **START**, aby uruchomić kamerę. Zezwól przeglądarce na dostęp.")

        # To jest poprawiony fragment w sekcji: elif app_mode == "📹 Kamera (DeepFace - Live/Foto)":

        # Definicja serwerów STUN (niezbędne, aby obraz przeszedł przez sieć w chmurze)
        RTC_CONFIGURATION = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }

        webrtc_streamer(
            key="emotion-filter",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionProcessor,
            rtc_configuration=RTC_CONFIGURATION,  # <--- DODANO TO
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        st.info("💡 Wideo może mieć opóźnienie, ponieważ analiza DeepFace jest wymagająca obliczeniowo.")

    else:
        # Tryb zdjęcia (Stary dobry camera_input)
        img_buffer = st.camera_input("Zrób zdjęcie")
        if img_buffer:
            temp = "temp.jpg"
            with open(temp, "wb") as f:
                f.write(img_buffer.getbuffer())

            col1, col2 = st.columns(2)
            col1.image(temp)

            with st.spinner("Analiza..."):
                try:
                    res = DeepFace.analyze(temp, actions=['emotion'], enforce_detection=False)
                    if isinstance(res, list): res = res[0]

                    all_emotions = res['emotion']
                    # Filtrowanie 3 emocji
                    scores = {k: all_emotions.get(k, 0) for k in TARGET_EMOTIONS.keys()}
                    total = sum(scores.values())

                    if total > 0:
                        norm_scores = {k: v / total for k, v in scores.items()}
                        winner = max(norm_scores, key=norm_scores.get)

                        col2.success(f"Wynik: **{TARGET_EMOTIONS[winner]}**")
                        col2.metric("Pewność", f"{norm_scores[winner]:.1%}")

                        # Wykres
                        df_chart = pd.DataFrame({
                            'Emocja': [TARGET_EMOTIONS[k] for k in norm_scores],
                            'Wynik': list(norm_scores.values())
                        })
                        fig = px.bar(df_chart, x='Emocja', y='Wynik', color='Emocja',
                                     color_discrete_map={'ZŁOŚĆ': '#FF4B4B', 'RADOŚĆ': '#2ECC71', 'SMUTEK': '#3498DB'})
                        col2.plotly_chart(fig)
                    else:
                        col2.warning("Wykryto twarz, ale żadna z emocji (Złość/Radość/Smutek) nie jest dominująca.")

                except Exception as e:
                    st.error(f"Błąd: {e}")