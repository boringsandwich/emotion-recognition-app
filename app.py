import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import random
import glob
import cv2  # Do kamery
from PIL import Image

# Importy do Twojego modelu (TensorFlow)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Import do nowej funkcji kamery (DeepFace)
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
DEEPFACE_TRANSLATION = {
    'angry': 'ZLOSC', 'disgust': 'OBRZYDZENIE', 'fear': 'STRACH',
    'happy': 'RADOSC', 'sad': 'SMUTEK', 'surprise': 'ZASKOCZENIE', 'neutral': 'NEUTRALNY'
}
# Kolory BGR dla OpenCV
DEEPFACE_COLORS = {
    'ZLOSC': (0, 0, 255), 'RADOSC': (0, 255, 0), 'SMUTEK': (255, 0, 0),
    'NEUTRALNY': (200, 200, 200), 'ZASKOCZENIE': (0, 255, 255)
}


# --- FUNKCJE POMOCNICZE (TWÓJ MODEL) ---

@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path):
    try:
        # Twój model wymaga 48x48 grayscale
        img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
        img_array = img_to_array(img)
        img_array /= 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception:
        return None


def get_dataset_images(limit=None):
    search_path = os.path.join(DATASET_TEST_PATH, '**', '*.jpg')
    found_files = glob.glob(search_path, recursive=True)
    if not found_files: return []
    if limit is not None and len(found_files) > limit:
        return random.sample(found_files, limit)
    return found_files


# =========================================================
# GŁÓWNY INTERFEJS (SELEKCJA TRYBU)
# =========================================================

st.sidebar.title("🎛️ Panel Sterowania")
app_mode = st.sidebar.selectbox(
    "Wybierz tryb aplikacji:",
    ["📂 Badanie (Mój Model - Oryginał)", "📹 Kamera (Live/Foto - DeepFace)"]
)

# ---------------------------------------------------------
# TRYB 1: BADANIE (To jest Twój stary kod w 100%)
# ---------------------------------------------------------
if app_mode == "📂 Badanie (Mój Model - Oryginał)":

    st.title("🧠 Inteligentny System Rozpoznawania Emocji (Twój Model)")

    # 1. Ładowanie Modelu
    model = load_ai_model()

    if model is None:
        st.error(f"Nie znaleziono modelu '{MODEL_PATH}'. Uruchom najpierw trening (emotion.py)!")
        st.stop()

    st.sidebar.header("⚙️ Źródło Danych (Badanie)")
    source_option = st.sidebar.radio(
        "Skąd pobrać zdjęcia?",
        ("📂 Własny folder (test_real)", "📚 Zbiór testowy (Dataset)")
    )

    image_paths = []
    current_source_name = ""

    # Logika wyboru plików (jak w oryginale)
    if source_option == "📂 Własny folder (test_real)":
        folder_path = st.sidebar.text_input("Ścieżka do folderu:", value=DEFAULT_USER_FOLDER)
        current_source_name = "Moje zdjęcia"
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_paths = [os.path.join(folder_path, f) for f in files]
        else:
            st.sidebar.warning("Folder nie istnieje.")

    else:  # Dataset
        current_source_name = "Zbiór Testowy (Dataset)"
        if os.path.exists(DATASET_TEST_PATH):
            load_all = st.sidebar.checkbox("Wczytaj WSZYSTKIE dostępne zdjęcia", value=False)
            if load_all:
                image_paths = get_dataset_images(limit=None)
                st.sidebar.warning(f"⚠️ Uwaga: Wczytano {len(image_paths)} zdjęć.")
            else:
                sample_size = st.sidebar.number_input("Liczba losowych zdjęć:", min_value=1, value=200, step=50)
                image_paths = get_dataset_images(limit=sample_size)
                st.sidebar.info(f"Pobrano losowe {len(image_paths)} zdjęć.")
        else:
            st.sidebar.error(f"Nie znaleziono folderu {DATASET_TEST_PATH}.")

    if not image_paths:
        st.warning("Brak zdjęć do analizy.")
        st.stop()

    # --- ANALIZA DANYCH (Zachowana logika sesji) ---
    session_key = f'analysis_{current_source_name}_{len(image_paths)}'

    if session_key not in st.session_state:
        st.session_state[session_key] = None

    if st.session_state[session_key] is None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        has_ground_truth = False
        total_imgs = len(image_paths)

        with st.spinner(f"Przetwarzanie {total_imgs} zdjęć Twoim modelem..."):
            for i, path in enumerate(image_paths):
                if i % (max(1, total_imgs // 20)) == 0:
                    progress_bar.progress(i / total_imgs)
                    status_text.text(f"Analiza obrazu {i + 1}/{total_imgs}")

                processed_img = preprocess_image(path)
                if processed_img is not None:
                    # Predykcja Twoim modelem
                    pred = model.predict(processed_img, verbose=0)[0]
                    idx = np.argmax(pred)
                    label = CLASSES[idx]
                    confidence = np.max(pred)

                    filename = os.path.basename(path)
                    parent_folder = os.path.basename(os.path.dirname(path))
                    true_label_eng = parent_folder if parent_folder in CLASSES else None
                    true_label_pl = TRANSLATION.get(true_label_eng, "-")

                    if true_label_eng: has_ground_truth = True

                    results.append({
                        "Plik": filename,
                        "Prawdziwa_Etykieta": true_label_pl,
                        "Wykryta_Emocja": TRANSLATION[label],
                        "Pewność": confidence,
                        "Raw_Angry": pred[0], "Raw_Happy": pred[1], "Raw_Sad": pred[2],
                        "Ścieżka": path,
                        "Poprawne": true_label_pl == TRANSLATION[label] if true_label_eng else None
                    })
            progress_bar.progress(1.0)
            status_text.empty()

        st.session_state[session_key] = pd.DataFrame(results)
        st.session_state[f"{session_key}_has_gt"] = has_ground_truth

    df = st.session_state[session_key]
    has_gt = st.session_state.get(f"{session_key}_has_gt", False)

    # --- ZAKŁADKI (Zachowane w 100%) ---
    tab1, tab2, tab3 = st.tabs(["📊 Raporty i Statystyki", "🔍 Przeglądarka (Interakcja)", "🧮 Wnioskowanie Bayesowskie"])

    # TAB 1: RAPORTY
    with tab1:
        st.header(f"Raport dla: {current_source_name}")
        with st.expander("📈 Zobacz historię treningu"):
            if os.path.exists("wykresy_treningu.png"):
                st.image("wykresy_treningu.png", use_container_width=True)
            else:
                st.warning("Brak pliku wykresy_treningu.png")

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(df, names='Wykryta_Emocja', title='Rozkład Wykrytych Emocji',
                             color='Wykryta_Emocja', color_discrete_map={v: COLORS[k] for k, v in TRANSLATION.items()})
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            avg_conf = df.groupby('Wykryta_Emocja')['Pewność'].mean().reset_index()
            fig_bar = px.bar(avg_conf, x='Wykryta_Emocja', y='Pewność', title='Średnia Pewność',
                             color='Wykryta_Emocja', color_discrete_map={v: COLORS[k] for k, v in TRANSLATION.items()})
            fig_bar.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(df[['Plik', 'Prawdziwa_Etykieta', 'Wykryta_Emocja', 'Pewność']], use_container_width=True)

    # TAB 2: PRZEGLĄDARKA
    with tab2:
        st.header("Interaktywna Przeglądarka")
        # Sortowanie: Najpierw poprawne, potem pewność
        df_sorted = df.sort_values(by=['Poprawne', 'Pewność'], ascending=[False, False], na_position='last')

        c1, c2 = st.columns(2)
        target = c1.selectbox("Filtruj po emocji:", ["Wszystkie"] + list(TRANSLATION.values()))
        amount = c2.slider("Liczba zdjęć:", 1, len(df), min(20, len(df)))

        if target == "Wszystkie":
            filtered_df = df_sorted.head(amount)
        else:
            filtered_df = df_sorted[df_sorted['Wykryta_Emocja'] == target].head(amount)

        cols = st.columns(5)
        for index, row in filtered_df.iterrows():
            with cols[index % 5]:
                st.image(row['Ścieżka'], use_container_width=True)
                # Kolorowanie wyniku
                color_style = "green" if row['Poprawne'] else "red"
                if row['Prawdziwa_Etykieta'] == "-": color_style = "black"

                st.markdown(f"**{row['Wykryta_Emocja']}** ({row['Pewność']:.0%})", unsafe_allow_html=True)

    # TAB 3: BAYES
    with tab3:
        st.header("Eksperyment: Wnioskowanie Bayesowskie")
        col_left, col_right = st.columns([1, 2])
        with col_left:
            sel_file = st.selectbox("Wybierz zdjęcie:", df['Plik'])
            row_data = df[df['Plik'] == sel_file].iloc[0]
            st.image(row_data['Ścieżka'], width=200)

            st.markdown("### Kontekst (Prior)")
            p_ang = st.slider("Złość (Kontekst)", 0.0, 1.0, 0.33)
            p_hap = st.slider("Radość (Kontekst)", 0.0, 1.0, 0.33)
            p_sad = st.slider("Smutek (Kontekst)", 0.0, 1.0, 0.33)

            priors = np.array([p_ang, p_hap, p_sad])
            if priors.sum() == 0: priors = np.ones(3)
            priors /= priors.sum()

        with col_right:
            likelihood = np.array([row_data['Raw_Angry'], row_data['Raw_Happy'], row_data['Raw_Sad']])
            posterior = likelihood * priors
            posterior /= posterior.sum()

            fig = go.Figure()
            emotions_list = list(TRANSLATION.values())
            fig.add_trace(go.Bar(x=emotions_list, y=likelihood, name='Model (Oczy)'))
            fig.add_trace(go.Bar(x=emotions_list, y=posterior, name='Bayes (Oczy + Kontekst)'))
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"Decyzja Bayesa: **{emotions_list[np.argmax(posterior)]}**")


# ---------------------------------------------------------
# TRYB 2: KAMERA (NOWOŚĆ - DeepFace)
# ---------------------------------------------------------
elif app_mode == "📹 Kamera (Live/Foto - DeepFace)":

    st.title("📹 Detekcja Emocji na Żywo")
    st.caption("Ten tryb używa silnika **DeepFace** (zewnętrzna biblioteka), aby działać płynnie na obrazie z kamery.")

    mode_cam = st.radio("Wybierz metodę:", ["🔴 Stream Video", "📸 Pojedyncze Zdjęcie"], horizontal=True)

    # --- PODTRYB: VIDEO LIVE ---
    if mode_cam == "🔴 Stream Video":
        col1, col2 = st.columns([3, 1])

        with col2:
            st.markdown("### Sterowanie")
            run = st.checkbox('🔴 Włącz Kamerę')
            st.info("System analizuje co 10. klatkę dla płynności.")
            text_placeholder = st.empty()

        with col1:
            frame_placeholder = st.image([])

        if run:
            # Używamy OpenCV do czytania kamery
            cap = cv2.VideoCapture(0)
            # Ładujemy detektor twarzy (szybki)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            frame_count = 0
            last_emotion = "Analiza..."
            last_color = (255, 255, 255)

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Nie można odczytać kamery.")
                    break

                # 1. Wykryj twarz (Haar Cascade)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    # Rysuj ramkę
                    cv2.rectangle(frame, (x, y), (x + w, y + h), last_color, 2)

                    # 2. Analizuj emocje (DeepFace) co 10 klatek
                    if frame_count % 10 == 0:
                        try:
                            # Wycinamy twarz
                            face_roi = frame[y:y + h, x:x + w]
                            # DeepFace robi magię
                            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                            # Obsługa wyniku (czasem jest to lista)
                            if isinstance(result, list): result = result[0]

                            emo_eng = result['dominant_emotion']
                            # Tłumaczenie na PL
                            last_emotion = DEEPFACE_TRANSLATION.get(emo_eng, emo_eng)
                            # Dobór koloru
                            last_color = DEEPFACE_COLORS.get(last_emotion, (255, 255, 255))

                            # Wyświetl tekst obok
                            text_placeholder.markdown(f"## Wykryto: **{last_emotion}**")

                        except Exception:
                            pass

                    # Podpisz ramkę
                    cv2.putText(frame, last_emotion, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, last_color, 2)

                # Konwersja BGR -> RGB dla Streamlit
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB")
                frame_count += 1

            cap.release()

    # --- PODTRYB: ZDJĘCIE ---
    else:
        st.subheader("Zrób zdjęcie i przeanalizuj")
        img_buffer = st.camera_input("Uśmiech!")

        if img_buffer is not None:
            # Zapisz tymczasowo
            temp_filename = "temp_snap.jpg"
            with open(temp_filename, "wb") as f:
                f.write(img_buffer.getbuffer())

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(temp_filename, caption="Twoje zdjęcie")

            with col_res2:
                with st.spinner("Analiza DeepFace..."):
                    try:
                        res = DeepFace.analyze(temp_filename, actions=['emotion'])
                        if isinstance(res, list): res = res[0]

                        emo = res['dominant_emotion']
                        emo_pl = DEEPFACE_TRANSLATION.get(emo, emo)
                        conf = res['emotion'][emo]

                        st.success(f"Emocja: **{emo_pl}**")
                        st.metric("Pewność", f"{conf:.1f}%")
                        st.json(res['emotion'])  # Pokaż wszystkie procenty
                    except Exception as e:
                        st.error(f"Nie wykryto twarzy. Spróbuj ponownie. ({e})")