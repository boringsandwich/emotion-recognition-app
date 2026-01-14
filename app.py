import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import random
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="System Analizy Emocji", layout="wide")

# Stałe
MODEL_PATH = 'moj_model_fer.h5'
DATASET_TEST_PATH = './dane_fer/test'  # Ścieżka do zbioru testowego z emotion.py
DEFAULT_USER_FOLDER = 'test_real'

CLASSES = ['angry', 'happy', 'sad']
TRANSLATION = {'angry': 'ZŁOŚĆ', 'happy': 'RADOŚĆ', 'sad': 'SMUTEK'}
COLORS = {'angry': '#FF4B4B', 'happy': '#2ECC71', 'sad': '#3498DB'}


# --- FUNKCJE POMOCNICZE ---

@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path):
    # Ładowanie i pre-processing
    try:
        img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
        img_array = img_to_array(img)
        img_array /= 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        return None


def get_dataset_images(limit=None):
    """
    Pobiera zdjęcia z folderu datasetu.
    limit=None oznacza pobranie WSZYSTKICH zdjęć.
    """
    search_path = os.path.join(DATASET_TEST_PATH, '**', '*.jpg')
    found_files = glob.glob(search_path, recursive=True)

    if not found_files:
        return []

    if limit is not None and len(found_files) > limit:
        return random.sample(found_files, limit)

    return found_files


# --- INTERFEJS UŻYTKOWNIKA ---

st.title("🧠 Inteligentny System Rozpoznawania Emocji")

# 1. Ładowanie Modelu
model = load_ai_model()

if model is None:
    st.error(f"Nie znaleziono modelu '{MODEL_PATH}'. Uruchom najpierw trening (emotion.py)!")
    st.stop()

# --- PASEK BOCZNY (WYBÓR ŹRÓDŁA) ---
st.sidebar.header("⚙️ Źródło Danych")

source_option = st.sidebar.radio(
    "Skąd pobrać zdjęcia?",
    ("📂 Własny folder (test_real)", "📚 Zbiór testowy (Dataset)")
)

image_paths = []
current_source_name = ""

if source_option == "📂 Własny folder (test_real)":
    folder_path = st.sidebar.text_input("Ścieżka do folderu:", value=DEFAULT_USER_FOLDER)
    current_source_name = "Moje zdjęcia"

    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths = [os.path.join(folder_path, f) for f in files]
    else:
        st.sidebar.warning("Folder nie istnieje.")

else:  # Opcja Dataset
    current_source_name = "Zbiór Testowy (Dataset)"
    if os.path.exists(DATASET_TEST_PATH):
        load_all = st.sidebar.checkbox("Wczytaj WSZYSTKIE dostępne zdjęcia", value=False)

        if load_all:
            image_paths = get_dataset_images(limit=None)
            st.sidebar.warning(f"⚠️ Uwaga: Wczytano {len(image_paths)} zdjęć. Analiza może chwilę potrwać!")
        else:
            sample_size = st.sidebar.number_input("Liczba losowych zdjęć:", min_value=1, value=200, step=50)
            image_paths = get_dataset_images(limit=sample_size)
            st.sidebar.info(f"Pobrano losowe {len(image_paths)} zdjęć.")
    else:
        st.sidebar.error(f"Nie znaleziono folderu {DATASET_TEST_PATH}. Czy rozpakowałeś dane?")

if not image_paths:
    st.warning("Brak zdjęć do analizy. Sprawdź ustawienia w panelu bocznym.")
    st.stop()

# --- ANALIZA DANYCH ---

session_key = f'analysis_{current_source_name}_{len(image_paths)}'

if session_key not in st.session_state:
    st.session_state[session_key] = None

if st.session_state[session_key] is None:
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    has_ground_truth = False
    total_imgs = len(image_paths)

    with st.spinner(f"Przetwarzanie {total_imgs} zdjęć..."):
        for i, path in enumerate(image_paths):
            if i % (max(1, total_imgs // 20)) == 0:
                progress_bar.progress(i / total_imgs)
                status_text.text(f"Analiza obrazu {i + 1}/{total_imgs}")

            processed_img = preprocess_image(path)
            if processed_img is not None:
                pred = model.predict(processed_img, verbose=0)[0]
                idx = np.argmax(pred)
                label = CLASSES[idx]
                confidence = np.max(pred)

                filename = os.path.basename(path)
                parent_folder = os.path.basename(os.path.dirname(path))
                true_label_eng = parent_folder if parent_folder in CLASSES else None
                true_label_pl = TRANSLATION.get(true_label_eng, "-")

                if true_label_eng:
                    has_ground_truth = True

                results.append({
                    "Plik": filename,
                    "Prawdziwa_Etykieta": true_label_pl,
                    "Wykryta_Emocja": TRANSLATION[label],
                    "Pewność": confidence,
                    "Raw_Angry": pred[0],
                    "Raw_Happy": pred[1],
                    "Raw_Sad": pred[2],
                    "Ścieżka": path,
                    "Poprawne": true_label_pl == TRANSLATION[label] if true_label_eng else None
                })

        progress_bar.progress(1.0)
        status_text.empty()

    st.session_state[session_key] = pd.DataFrame(results)
    st.session_state[f"{session_key}_has_gt"] = has_ground_truth

df = st.session_state[session_key]
has_gt = st.session_state.get(f"{session_key}_has_gt", False)

# --- ZAKŁADKI ---

tab1, tab2, tab3 = st.tabs(["📊 Raporty i Statystyki", "🔍 Przeglądarka (Interakcja)", "🧮 Wnioskowanie Bayesowskie"])

# --- TAB 1: RAPORTY ---
with tab1:
    st.header(f"Raport dla: {current_source_name}")

    with st.expander("📈 Zobacz historię treningu (Wykresy Accuracy/Loss)"):
        if os.path.exists("wykresy_treningu.png"):
            st.image("wykresy_treningu.png", caption="Przebieg uczenia modelu", use_container_width=True)
        else:
            st.warning("Brak pliku wykresy_treningu.png")

    if has_gt:
        st.markdown("---")
        st.subheader("Bilans Wyników (Test)")

        df_labeled = df.dropna(subset=['Poprawne'])
        total_labeled = len(df_labeled)
        correct_count = df_labeled['Poprawne'].sum()
        incorrect_count = total_labeled - correct_count
        accuracy = correct_count / total_labeled if total_labeled > 0 else 0

        st.markdown(f"""
            <div style="display: flex; justify-content: center; gap: 50px; margin-bottom: 20px;">
                <div style="text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;">
                    <div style="font-size: 4em; color: #2ECC71; font-weight: bold; line-height: 1;">{correct_count}</div>
                    <div style="font-size: 1.2em; color: #2ECC71; font-weight: bold; margin-top: 10px;">POPRAWNE ✅</div>
                </div>
                <div style="text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px;">
                    <div style="font-size: 4em; color: #FF4B4B; font-weight: bold; line-height: 1;">{incorrect_count}</div>
                    <div style="font-size: 1.2em; color: #FF4B4B; font-weight: bold; margin-top: 10px;">BŁĘDNE ❌</div>
                </div>
            </div>
            <div style="text-align: center; font-size: 1.1em; margin-bottom: 30px;">
                Łączna liczba próbek testowych: <strong>{total_labeled}</strong><br>
                Skuteczność modelu (Accuracy): <strong style="font-size: 1.3em; color: #333;">{accuracy:.1%}</strong>
            </div>
            <hr>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(df, names='Wykryta_Emocja', title='Rozkład Wykrytych Emocji',
                         color='Wykryta_Emocja', color_discrete_map={v: COLORS[k] for k, v in TRANSLATION.items()})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        avg_conf = df.groupby('Wykryta_Emocja')['Pewność'].mean().reset_index()
        fig_bar = px.bar(avg_conf, x='Wykryta_Emocja', y='Pewność', title='Średnia Pewność Modelu',
                         color='Wykryta_Emocja', color_discrete_map={v: COLORS[k] for k, v in TRANSLATION.items()})
        fig_bar.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(df[['Plik', 'Prawdziwa_Etykieta', 'Wykryta_Emocja', 'Pewność']].style.highlight_max(axis=0),
                 use_container_width=True)

# --- TAB 2: INTERAKCJA (SORTOWANIE I FILTROWANIE) ---
with tab2:
    st.header("Interaktywna Przeglądarka")
    st.markdown("Zdjęcia są sortowane w kolejności: **Poprawne (najpewniejsze) ➜ Błędne**.")

    # KROK 1: SORTOWANIE (KLUCZOWA ZMIANA)
    # Sortujemy najpierw po kolumnie 'Poprawne' (True > False), potem po 'Pewność' (Malejąco)
    # ascending=[False, False] oznacza: True jest wyżej niż False, a wyższa pewność wyżej niż niższa.
    df_sorted = df.sort_values(by=['Poprawne', 'Pewność'], ascending=[False, False], na_position='last')

    c1, c2 = st.columns(2)
    with c1:
        options = ["Wszystkie"] + list(TRANSLATION.values())
        target_emotion = st.selectbox("Pokaż mi zdjęcia, gdzie wykryto:", options)
    with c2:
        total_available = len(df)
        default_val = min(20, total_available)
        if total_available > 0:
            amount = st.slider("Liczba zdjęć do wyświetlenia (TOP):", 1, total_available, default_val)
        else:
            amount = 0
            st.write("Brak zdjęć.")

    # KROK 2: FILTROWANIE POSORTOWANEJ RAMKI
    if target_emotion == "Wszystkie":
        filtered_df = df_sorted.head(amount)
    else:
        filtered_df = df_sorted[df_sorted['Wykryta_Emocja'] == target_emotion].head(amount)

    if filtered_df.empty:
        st.info(f"Nie znaleziono zdjęć spełniających kryteria.")
    else:
        cols = st.columns(5)
        for index, row in filtered_df.iterrows():
            col_idx = index % 5
            with cols[col_idx]:
                st.image(row['Ścieżka'], use_container_width=True)

                detected = row['Wykryta_Emocja']
                confidence_str = f"({row['Pewność']:.1%})"
                truth = row['Prawdziwa_Etykieta']

                caption_html = f"**Wykryto:** {detected} {confidence_str}"

                if truth != "-":
                    color_style = "color: green; font-weight: bold;" if truth == detected else "color: red; font-weight: bold;"
                    caption_html += f"<br>Prawda: <span style='{color_style}'>{truth}</span>"

                st.markdown(caption_html, unsafe_allow_html=True)
                st.write("")

# --- TAB 3: BAYES ---
with tab3:
    st.header("Eksperyment: Wnioskowanie Bayesowskie")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        selected_file_name = st.selectbox("Wybierz zdjęcie do analizy:", df['Plik'])
        row_data = df[df['Plik'] == selected_file_name].iloc[0]

        st.image(row_data['Ścieżka'], width=200,
                 caption=f"Aktualny werdykt modelu: {row_data['Wykryta_Emocja']} ({row_data['Pewność']:.1%})")

        st.markdown("### Kontekst Sytuacyjny (Prior)")
        prior_angry = st.slider("Napięcie / Złość", 0.0, 1.0, 0.33)
        prior_happy = st.slider("Radość / Impreza", 0.0, 1.0, 0.33)
        prior_sad = st.slider("Smutek / Żałoba", 0.0, 1.0, 0.33)

        total = prior_angry + prior_happy + prior_sad
        if total == 0: total = 1
        priors = np.array([prior_angry, prior_happy, prior_sad]) / total

    with col_right:
        likelihood = np.array([row_data['Raw_Angry'], row_data['Raw_Happy'], row_data['Raw_Sad']])

        posterior = likelihood * priors
        if np.sum(posterior) > 0:
            posterior /= np.sum(posterior)

        fig = go.Figure()
        emotions_pl = list(TRANSLATION.values())

        fig.add_trace(go.Bar(x=emotions_pl, y=likelihood, name='Model (Wzrok)', marker_color='#95a5a6'))
        fig.add_trace(go.Bar(x=emotions_pl, y=posterior, name='Wynik Bayesowski', marker_color='#8e44ad'))

        fig.update_layout(barmode='group', title="Wpływ kontekstu na decyzję", height=400)
        st.plotly_chart(fig, use_container_width=True)

        winner_idx = np.argmax(posterior)
        final_verdict = emotions_pl[winner_idx]
        final_conf = posterior[winner_idx]

        st.success(f"Ostateczna decyzja systemu: **{final_verdict}** ({final_conf:.1%})")

        model_winner = emotions_pl[np.argmax(likelihood)]
        if model_winner != final_verdict:
            st.error(
                f"😮 Kontekst zmienił wynik! Model widział **{model_winner}**, ale sytuacja wskazuje na **{final_verdict}**.")