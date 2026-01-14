import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
import sys

# --- KONFIGURACJA ---
zip_path = 'archive.zip'
extract_to = './dane_fer'

# --- KROK 1: Rozpakowanie ---
if not os.path.exists(zip_path) and not os.path.exists(extract_to):
    print(f"BŁĄD: Nie widzę pliku '{zip_path}'.")
    sys.exit()

if not os.path.exists(extract_to):
    print(f"Rozpakowywanie '{zip_path}'...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
else:
    print(f"Folder '{extract_to}' istnieje.")

# --- KROK 2: Ścieżki ---
found_train = None
found_test = None
for root, dirs, files in os.walk(extract_to):
    if 'train' in dirs: found_train = os.path.join(root, 'train')
    if 'test' in dirs: found_test = os.path.join(root, 'test')

if not found_train or not found_test:
    print("BŁĄD: Nie znaleziono folderów train/test.")
    sys.exit()

# --- KROK 3: Wybór klas ---
available = os.listdir(found_train)
target_classes = sorted([x for x in available if x.lower() in ['angry', 'happy', 'sad']])
print(f"Klasy: {target_classes}")

if len(target_classes) != 3:
    print("UWAGA: Nie znaleziono 3 wymaganych emocji.")

# --- KROK 4: Generatory ---
datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    found_train,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    classes=target_classes,
    shuffle=True
)

test_generator = datagen.flow_from_directory(
    found_test,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    classes=target_classes,
    shuffle=False  # Ważne: False, żebyśmy mogli porównać wyniki z prawdą
)

# --- KROK 5: Model ---
model = Sequential([
    Input(shape=(48, 48, 1)),

    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(len(target_classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- KROK 6: Trening ---
print(f"\nRozpoczynam trening (80 epok)...")
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=test_generator
)

# --- KROK 7: Zapisywanie wykresów treningu ---
print("\nGenerowanie wykresów treningu...")
plt.figure(figsize=(12, 5))

# Dokładność
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title('Dokładność (Accuracy)')
plt.xlabel('Epoka')
plt.legend()
plt.grid(True)

# Strata
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Strata')
plt.plot(history.history['val_loss'], label='Test')
plt.title('Błąd (Loss)')
plt.xlabel('Epoka')
plt.legend()
plt.grid(True)

filename_charts = 'wykresy_treningu.png'
plt.savefig(filename_charts)
print(f"Zapisano wykresy do pliku: {filename_charts}")
# plt.show() # Opcjonalnie odkomentuj, jeśli chcesz widzieć okno

# --- KROK 8: Przygotowanie danych do analizy (Pobranie obrazów testowych) ---
print("\nPobieranie danych testowych do analizy błędów...")

# Musimy wyciągnąć obrazy i etykiety z generatora
# Ponieważ generator podaje dane w partiach (batchach), musimy je skleić
test_images = []
test_labels = []

# Reset generatora na początek
test_generator.reset()

# Iterujemy przez cały zbiór testowy
for i in range(len(test_generator)):
    batch_x, batch_y = next(test_generator)
    test_images.append(batch_x)
    test_labels.append(batch_y)

# Sklejamy w jedną dużą tablicę (numpy array)
X_test_all = np.concatenate(test_images)
y_test_all = np.concatenate(test_labels)

# Robimy predykcję dla całego zbioru
predictions = model.predict(X_test_all)

# Zamieniamy one-hot encoding (np. [0, 1, 0]) na numer klasy (np. 1)
y_pred_classes = np.argmax(predictions, axis=1)
y_true_classes = np.argmax(y_test_all, axis=1)

# Słownik nazw klas (np. 0 -> 'angry')
class_labels = {v: k for k, v in test_generator.class_indices.items()}

# Znajdujemy indeksy dobrych i złych odpowiedzi
correct_indices = np.where(y_pred_classes == y_true_classes)[0]
incorrect_indices = np.where(y_pred_classes != y_true_classes)[0]

print(f"Liczba poprawnych: {len(correct_indices)}")
print(f"Liczba błędnych: {len(incorrect_indices)}")


# --- KROK 9: Wizualizacja i zapis (Funkcja pomocnicza) ---

def save_visualizations(indices, filename, title_prefix, num_images=20):
    if len(indices) < num_images:
        num_images = len(indices)

    # Wybieramy losowe 20 indeksów z podanej listy
    selected_indices = np.random.choice(indices, num_images, replace=False)

    plt.figure(figsize=(15, 12))  # Rozmiar obrazka
    rows = 4
    cols = 5

    for i, idx in enumerate(selected_indices):
        plt.subplot(rows, cols, i + 1)

        # Wyświetlamy obraz (48x48)
        img = X_test_all[idx].reshape(48, 48)
        plt.imshow(img, cmap='gray')

        true_label = class_labels[y_true_classes[idx]]
        pred_label = class_labels[y_pred_classes[idx]]

        # Kolor tekstu: zielony jeśli dobrze, czerwony jeśli źle
        color = 'green' if true_label == pred_label else 'red'

        plt.title(f"Prawda: {true_label}\nModel: {pred_label}", color=color, fontsize=10)
        plt.axis('off')

    plt.suptitle(f"{title_prefix} - Przykłady", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Zapisano plik: {filename}")
    plt.close()  # Zamykamy, żeby zwolnić pamięć


# Generujemy plik z poprawnymi
save_visualizations(correct_indices, 'poprawne_przyklady.png', "DOBRE PREDYKCJE")

# Generujemy plik z błędnymi
save_visualizations(incorrect_indices, 'bledne_przyklady.png', "BŁĘDNE PREDYKCJE")

# --- TUTAJ JEST DODANE ZAPISYWANIE MODELU ---
print("\nZapisuję wytrenowany model do pliku...")
model.save('moj_model_fer.h5')
print("SUKCES: Model zapisany jako 'moj_model_fer.h5'")

print("\n--- KONIEC ---")
print("Sprawdź folder projektu.")
print("Powinny być tam 3 nowe pliki PNG (wykresy i przykłady) oraz plik modelu .h5")

