import numpy as np
import pandas as pd
import parselmouth
import sounddevice as sd
import scipy.io.wavfile as wav
import tkinter as tk
from tkinter import messagebox, PhotoImage
import os
import speech_recognition as sr
from transformers import pipeline
import pyttsx3, random, pygame
import mysql.connector
from PIL import Image, ImageTk, ImageEnhance,ImageSequence
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import hashlib,threading

# --------------------
# Global Variables for Login/Tracking
# --------------------
user_name = None   # Will store the user's full (spoken) name.
user_id = None
user_email = None  # Optionally store the email if needed

# --------------------
# Initialize text-to-speech engine
# --------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 140)  # Set speech rate
# Adjust volume (0.0 to 1.0)
tts_engine.setProperty('volume', 1.5)  # Max volume
voices = tts_engine.getProperty('voices')
tts_engine.setProperty('voice', voices[0].id)  # Switch to Zira (Female) for a more emotional tone

# --------------------
# Load NLP models
# --------------------
try:
    tone_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# --------------------
# KNN Model for stress and depression prediction based on pitch features
# --------------------
knn_model = KNeighborsClassifier(n_neighbors=3)

# --------------------
# Training and Data Functions
# --------------------
def load_training_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data[['mean_pitch', 'pitch_variability']].values
    y = data[['stress_level', 'depression_level']].values
    return X, y

def train_knn_model():
    csv_file = 'mental_health_dataset.csv'  # Replace with your CSV file path
    X_train, y_train = load_training_data(csv_file)
    knn_model.fit(X_train, y_train)
    print("KNN Model trained successfully with CSV data.")

train_knn_model()

# --------------------
# Audio and Analysis Functions
# --------------------
def record_audio(duration=5, sample_rate=44100):
    """
    Records audio for a specified duration and returns the audio data along with the sample rate.
    """
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    return audio_data, sample_rate


def save_audio_to_wav(audio_data, sample_rate, filename="recorded_audio.wav"):
    wav.write(filename, sample_rate, audio_data)
    return filename

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
            return transcription
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Speech Recognition error: {e}")
        return ""

def analyze_pitch(audio_file):
    try:
        sound = parselmouth.Sound(audio_file)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        return pitch_values
    except Exception as e:
        print(f"Error analyzing pitch: {e}")
        return np.array([])

def extract_pitch_features(pitch_values):
    valid_pitches = [p for p in pitch_values if p > 0]
    if not valid_pitches:
        return np.array([0, 0])
    return np.array([np.mean(valid_pitches), np.std(valid_pitches)])

def analyze_tone_emotion_stress(text, pitch_values, user_id):
    tone_result = tone_analyzer(text)
    emotion_result = emotion_analyzer(text)

    tone_label = tone_result[0]['label']
    tone_score = tone_result[0]['score']
    emotion_label = emotion_result[0][0]['label']
    emotion_score = emotion_result[0][0]['score']

    pitch_features = extract_pitch_features(pitch_values)
    prediction = knn_model.predict([pitch_features])
    stress_level = "High Stress" if prediction[0][0] == 1 else "Low Stress"
    depression_level = "High Depression" if prediction[0][1] == 1 else "Low Depression"

    store_analysis_results(user_id, tone_label, tone_score, emotion_label, emotion_score,
                           stress_level, depression_level, *pitch_features)

    result_str = (f"Tone: {tone_label} ({tone_score:.2f}), Emotion: {emotion_label} ({emotion_score:.2f}),\n"
                  f"Stress: {stress_level}, Depression: {depression_level}")
    return result_str

def capture_audio_input(prompt):
    print(prompt)
    tts_engine.say(prompt)
    tts_engine.runAndWait()

    audio_data, sample_rate = record_audio(duration=5)
    audio_file = save_audio_to_wav(audio_data, sample_rate, filename="temp_audio_input.wav")
    transcription = transcribe_audio(audio_file)

    if not transcription:
        print("Could not capture your input. Please try again.")
        tts_engine.say("Could not capture your input. Please try again.")
        tts_engine.runAndWait()
        return capture_audio_input(prompt)  # Retry

    print(f"You said: {transcription}")
    return transcription

# --------------------
# Database Interactions
# --------------------
def check_user_exists(email):
    """Return a tuple (user_id, name) if a user with the given email exists."""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="lalima@29",  # Replace with your MySQL password
            database="data"
        )
        cursor = conn.cursor(buffered=True)
        query = "SELECT user_id, name FROM user_info WHERE email = %s"
        cursor.execute(query, (email,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return None

def store_user_details(name, age, gender, email, address):
    try:
        conn = mysql.connector.connect(host="localhost", user="root", password="lalima@29", database="data")
        cursor = conn.cursor()
        query = """INSERT INTO user_info (name, age, gender, email, address) 
                   VALUES (%s, %s, %s, %s, %s)"""
        cursor.execute(query, (name, age, gender, email, address))
        conn.commit()
        user_id_local = cursor.lastrowid
        cursor.close()
        conn.close()
        return user_id_local
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return None

def update_user_details(user_id, sleeping_hours, fav_artist, substance_use):
    try:
        conn = mysql.connector.connect(host="localhost", user="root", password="lalima@29", database="data")
        cursor = conn.cursor()
        update_query = """
        UPDATE user_additional_details
        SET sleeping_hours = %s, fav_artist = %s, substance_use = %s
        WHERE user_id = %s
        """
        cursor.execute(update_query, (sleeping_hours, fav_artist, substance_use, user_id))
        conn.commit()
        cursor.close()
        conn.close()
        return "User details updated successfully."
    except mysql.connector.Error as err:
        print(f"Error updating details: {err}")
        return f"Error: {err}"

def store_analysis_results(user_id, tone_label, tone_score, emotion_label, emotion_score,
                           stress_level, depression_level, pitch_mean, pitch_variability):
    try:
        conn = mysql.connector.connect(host="localhost", user="root", password="lalima@29", database="data")
        cursor = conn.cursor()
        query = """INSERT INTO user_analysis (user_id, tone_label, tone_score, emotion_label, emotion_score, 
                   stress_level, depression_level, pitch_mean, pitch_variability)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(query, (user_id, tone_label, tone_score, emotion_label, emotion_score,
                                 stress_level, depression_level, pitch_mean, pitch_variability))
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        print(f"Database error: {e}")

# --------------------
# User Detail and Daily Update Flow
# --------------------
def ask_for_details():
    global user_id, user_name
    # If the user is already logged in, we already have their full name.
    if user_name is not None and user_id is not None:
        print(f"Welcome back, {user_name}. Checking for daily updates.")
        tts_engine.say(f"Welcome back, {user_name}. Checking for daily updates.")
        tts_engine.runAndWait()
        check_and_update_details(user_id)
    else:
        def get_audio_input(prompt):
            print(prompt)
            tts_engine.say(prompt)
            tts_engine.runAndWait()
            audio_data, sample_rate = record_audio(duration=5)
            audio_file = save_audio_to_wav(audio_data, sample_rate, filename="temp_audio_input.wav")
            transcription = transcribe_audio(audio_file)
            if not transcription:
                print("Could not capture your input. Please try again.")
                tts_engine.say("Could not capture your input. Please try again.")
                tts_engine.runAndWait()
                return get_audio_input(prompt)
            print(f"You said: {transcription}")
            return transcription

        # For new users, ask for details.
        user_name = get_audio_input("Please say your name.")
        email = get_audio_input("Please spell out your email address.")
        age = get_audio_input("Please say your age.")
        gender = get_audio_input("Please say your gender.")
        address = get_audio_input("Please say your address.")
        user_id = store_user_details(user_name, age, gender, email, address)
        print(f"User details stored. User ID: {user_id}")
        tts_engine.say("Thank you. Your details have been stored. Starting your analysis now.")
        tts_engine.runAndWait()
        check_and_update_details(user_id)

def check_and_update_details(user_id):
    try:
        conn = mysql.connector.connect(host="localhost", user="root", password="lalima@29", database="data")
        cursor = conn.cursor()
        query = "SELECT update_date FROM user_additional_details WHERE user_id = %s ORDER BY update_date DESC LIMIT 1"
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        current_date = pd.to_datetime("today").date()
        if result:
            last_update_date = result[0]
            # Ensure the date is in proper format for comparison
            if isinstance(last_update_date, pd.Timestamp):
                last_update_date = last_update_date.date()
            if last_update_date != current_date:
                print("It's time for your daily update session.")
                tts_engine.say("It's time for your daily update session.")
                tts_engine.runAndWait()
                ask_additional_details(user_id)
            else:
                # If the user already updated details today, proceed directly to analysis.
                start_analysis()
        else:
            # No update found for this user yet.
            ask_additional_details(user_id)
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
def ask_additional_details(user_id):
    # Ask for sleeping hours with confirmation and follow-up.
    sleeping_hours = capture_audio_input(f"{user_name}, how many hours do you sleep each night?")
    confirmation = capture_audio_input(f"You said {sleeping_hours} hours. Is that correct? Please say yes or no.")
    if confirmation.lower() not in ['yes', 'yeah', 'correct']:
        sleeping_hours = capture_audio_input("Okay, please tell me again, how many hours do you sleep each night?")
    
    sleep_feeling = capture_audio_input("Do you feel well-rested with that amount of sleep? Please say yes or no.")

    # Ask about favorite artist with confirmation and follow-up.
    fav_artist = capture_audio_input(f"{user_name}, who is your favorite artist?")
    confirmation = capture_audio_input(f"You said your favorite artist is {fav_artist}. Is that correct? Please say yes or no.")
    if confirmation.lower() not in ['yes', 'yeah', 'correct']:
        fav_artist = capture_audio_input("Alright, please tell me again, who is your favorite artist?")
    
    fav_song = capture_audio_input(f"Do you have a favorite song by {fav_artist}? If yes, please name it, otherwise say no.")

    # Ask about substance use with confirmation.
    substance_use = capture_audio_input(f"{user_name}, do you use any substances like drugs or alcohol? "
                                        "You can simply say yes or no, or provide more details if you prefer.")
    confirmation = capture_audio_input(f"You mentioned: {substance_use}. Is that correct? Please say yes or no.")
    if confirmation.lower() not in ['yes', 'yeah', 'correct']:
        substance_use = capture_audio_input("Please tell me again about your substance use.")

    # Offer an option for additional lifestyle details.
    additional_response = capture_audio_input("Would you like to add any more details about your lifestyle? Please say yes or no.")
    extra_detail = None
    if additional_response.lower() in ['yes', 'yeah']:
        extra_detail = capture_audio_input("Please give the feedback about this session")
    
    # Confirm final inputs before proceeding.
    summary = (
        f"Here's what I've gathered:\n"
        f"Sleeping Hours: {sleeping_hours} hours (Rested: {sleep_feeling})\n"
        f"Favorite Artist: {fav_artist} (Favorite Song: {fav_song})\n"
        f"Substance Use: {substance_use}\n"
    )
    if extra_detail:
        summary += f"Additional Details: {extra_detail}\n"
    
    summary += "Is this information correct? Please say yes or no."
    final_confirmation = capture_audio_input(summary)
    if final_confirmation.lower() not in ['yes', 'yeah', 'correct']:
        tts_engine.say("Let's try entering your details again.")
        tts_engine.runAndWait()
        return ask_additional_details(user_id)

    # Optionally, play a message before saving details.
    tts_engine.say(f"{user_name}, this session is finished now !")
    tts_engine.runAndWait()

    # Save the additional details to the database.
    try:
        conn = mysql.connector.connect(host="localhost", user="root", password="lalima@29", database="data")
        cursor = conn.cursor()
        # Adjust the INSERT query to match your database schema. Here we assume extra columns for extra details.
        query = """
            INSERT INTO user_additional_details (
                user_id, sleeping_hours, sleep_feeling, fav_artist, fav_song, substance_use, extra_detail, update_date
            ) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        update_date = pd.to_datetime("today").date()
        cursor.execute(query, (user_id, sleeping_hours, sleep_feeling, fav_artist, fav_song, substance_use, extra_detail, update_date))
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
    
    start_analysis()

# --------------------
# Analysis and Graph Display
# --------------------
def start_analysis():
    for widget in root.winfo_children():
        widget.destroy()
    # Now the GIF is removed, and you can set up the analysis view.
    root.geometry("900x600")
    
    print("I am going to start analysis.")
    tts_engine.say("its time for your analysis session. I am going to start analysis.")
    tts_engine.runAndWait()
    
    audio_data, sample_rate = record_audio()
    audio_file = save_audio_to_wav(audio_data, sample_rate)
    pitch_values = analyze_pitch(audio_file)
    transcription = transcribe_audio(audio_file)
    result = analyze_tone_emotion_stress(transcription, pitch_values, user_id)
    
    print("Analysis is finished.")
    tts_engine.say("Analysis is finished.")
    tts_engine.runAndWait()
    tts_engine.say(result)
    tts_engine.runAndWait()
    
    # Update result_label (if needed) or create a new label for the result
    result_label = tk.Label(root, text=result, font=("Maiandra GD", 14), bg="light blue")
    result_label.pack(pady=10)
    
    pitch_means = get_pitch_mean_from_db(user_id)
    if pitch_means:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(pitch_means, marker='o', linestyle='-', color='b')
        ax.set_title("Pitch Mean Over Time")
        ax.set_xlabel("Analysis #")
        ax.set_ylabel("Pitch Mean (Hz)")
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
    else:
        no_data_label = tk.Label(root, text="No pitch data available to plot.", font=("Maiandra GD", 12))
        no_data_label.pack(pady=10)
    
    user_report_button = tk.Button(root, text="User Report", font=("Maiandra GD", 14),
                                   command=lambda: show_user_report_in_main(result, pitch_means))
    user_report_button.pack(pady=20)

def get_pitch_mean_from_db(user_id):
    try:
        conn = mysql.connector.connect(host="localhost", user="root", password="lalima@29", database="data")
        cursor = conn.cursor()
        query = "SELECT pitch_mean FROM user_analysis WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return [result[0] for result in results]
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return []

def get_user_info(user_id):
    try:
        conn = mysql.connector.connect(host="localhost", user="root", password="lalima@29", database="data")
        cursor = conn.cursor()
        query = "SELECT name, age, gender, email, address FROM user_info WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return None

def get_last_update_date(user_id):
    try:
        conn = mysql.connector.connect(host="localhost", user="root", password="lalima@29", database="data")
        cursor = conn.cursor()
        query = "SELECT update_date FROM user_additional_details WHERE user_id = %s ORDER BY update_date DESC LIMIT 1"
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            return result[0]
        else:
            return "N/A"
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return "N/A"

def doctor_details():
    # Create a new window for doctor details.
    doctor_window = tk.Toplevel()
    doctor_window.title("Doctor Details")
    doctor_window.geometry("750x400")
    doctor_window.configure(bg="light blue")  # Set the window background color to light blue
    
    doctor_info = (
        "Dr. M. Sree Prathap\n"
        "Specialist in Psychiatrist\n"
        "Clinic: Shadithya Hospital\n"
        "Phone: +91 44 22640745, +91 8939729999, +91 7823990238\n"
        "Email: admin@shadithyahospitals.com\n"
        "Address:No: 7, Tannery St, Pallavaram, Chennai, Tamil Nadu 600043\n"
        "Consultation Hours: Please contact the hospital directly to confirm consultation hours.\n"
        "For appointments or inquiries, you can contact the clinic via phone or email."
    )
    label = tk.Label(doctor_window, text=doctor_info, font=("Maiandra GD", 14), justify="left",bg="light blue")
    label.pack(pady=20, padx=20)
    
    close_button = tk.Button(doctor_window, text="Close", font=("Maiandra GD", 14), command=doctor_window.destroy)
    close_button.pack(pady=10)

def ask_confirmation():
    # Create the confirmation window associated with the main window
    confirmation_window = tk.Toplevel(root)
    confirmation_window.title("Choose an Option")
    confirmation_window.geometry("450x400")
    confirmation_window.configure(bg="light blue")
    
    prompt_label = tk.Label(confirmation_window, text="What would you like to do?", font=("Maiandra GD", 14), bg="light blue")
    prompt_label.pack(pady=20)
    
    # Example Option buttons
    game_button = tk.Button(
        confirmation_window,
        text="Play Default Game",
        font=("Maiandra GD", 14),
        command=lambda: [
            create_memory_match_game(
                tk.Toplevel(root),  # Ensure new window is linked to root
                r"C:\Users\lalim\OneDrive\Desktop\temp\images\game.mp3", 
                r"C:\Users\lalim\OneDrive\Desktop\temp\images"
            ),
            confirmation_window.destroy()
        ]
    )
    game_button.pack(pady=5)
    
    meditation_button = tk.Button(
      confirmation_window,
      text="Start Meditation",
      font=("Maiandra GD", 14),
    command=lambda: [
        play_meditation_music(
            tk.Toplevel(root),  # Ensure new window is linked to root
            r"C:\Users\lalim\OneDrive\Desktop\temp\images\med.mp3",
            r"C:\Users\lalim\OneDrive\Desktop\temp\images\confused.gif"  # Pass the GIF file
        ),
        confirmation_window.destroy()
    ]
)

    meditation_button.pack(pady=5)
    
    doctor_button = tk.Button(
        confirmation_window,
        text="Doctor Details",
        font=("Maiandra GD", 14),
        command=lambda: [
            doctor_details(),
            confirmation_window.destroy()
        ]
    )
    doctor_button.pack(pady=5)
    
    cancel_button = tk.Button(
        confirmation_window,
        text="Cancel",
        font=("Maiandra GD", 14),
        command=confirmation_window.destroy
    )
    cancel_button.pack(pady=5)

# In your show_user_report_in_main (or similar) function, add the buttons:
def show_user_report_in_main(report_text, pitch_means):
    root.geometry("900x800")
    for widget in root.winfo_children():
        widget.destroy()
    user_info = get_user_info(user_id)
    last_update = get_last_update_date(user_id)
    if user_info:
        info_text = (f"Name: {user_info[0]}\n"
                     f"Age: {user_info[1]}\n"
                     f"Gender: {user_info[2]}\n"
                     f"Email: {user_info[3]}\n"
                     f"Address: {user_info[4]}\n")
    else:
        info_text = "User details not found.\n"
    info_text += f"\nLast Update: {last_update}\n\nAnalysis Result:\n{report_text}"
    report_label = tk.Label(root, text=info_text, font=("Maiandra GD", 14), justify="left", wraplength=500,bg="light blue")
    report_label.pack(pady=10)
    if pitch_means:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(pitch_means, marker='o', linestyle='-', color='b')
        ax.set_title("Pitch Mean Over Time")
        ax.set_xlabel("Analysis #")
        ax.set_ylabel("Pitch Mean (Hz)")
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
    else:
        no_data_label = tk.Label(root, text="No pitch data available to plot.", font=("Maiandra GD", 12))
        no_data_label.pack(pady=10)
   # Create a frame to hold both buttons
    # Create a frame to hold both buttons
    button_frame = tk.Frame(root, bg="light blue")
    button_frame.pack(pady=10)
    # Suggestion Button (should trigger ask_confirmation)
    suggestion_button = tk.Button(button_frame, text="Suggestion", command=ask_confirmation, font=("Maiandra GD", 14))
    suggestion_button.pack(side="left", padx=10)

    # Exit Button (closes the application)
    exit_button = tk.Button(button_frame, text="Exit", command=lambda: root.destroy(), fg="white", bg="red", font=("Maiandra GD", 14))
    exit_button.pack(side="left", padx=10)


# --------------------
# Memory Match Game and Meditation Functions
# --------------------
def create_memory_match_game(game_root, bg_music, image_dir):
    pygame.mixer.init()
    pygame.mixer.music.load(bg_music)
    pygame.mixer.music.play(-1)
    def enhance_image(image):
        if image.mode == "P":
            image = image.convert("RGBA")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(2.0)
    def resize_image(image, target_size=(80, 80)):
        return image.resize(target_size, Image.LANCZOS)
    def load_images(image_dir):
        images, missing_images = [], []
        for i in range(1, 7):
            image_path = os.path.join(image_dir, f"image{i}.png")
            if os.path.exists(image_path):
                image = Image.open(image_path)
                image = resize_image(enhance_image(image))
                images.append(ImageTk.PhotoImage(image))
            else:
                missing_images.append(image_path)
                images.append(None)
        if missing_images:
            with open("missing_images.txt", "w") as f:
                f.writelines(f"{path}\n" for path in missing_images)
        return images
    def load_hidden_image(image_dir):
        hidden_image_path = os.path.join(image_dir, "hidden.png")
        if os.path.exists(hidden_image_path):
            hidden_image = Image.open(hidden_image_path)
            return ImageTk.PhotoImage(resize_image(hidden_image))
        return PhotoImage(width=80, height=80)
    images = load_images(image_dir)
    hidden_image = load_hidden_image(image_dir)
    cards = list(images * 2)
    random.shuffle(cards)
    revealed_cards = []
    matches = 0
    moves = 0
    game_result_label = tk.Label(game_root, text="", font=("Maiandra GD", 14), fg="green")
    game_result_label.grid(row=5, column=0, columnspan=4, pady=10)
    def on_card_click(index, button):
        nonlocal matches, moves
        if button["image"] == str(hidden_image) and cards[index]:
            button.config(image=cards[index])
            revealed_cards.append((index, button))
            if len(revealed_cards) == 2:
                moves += 1
                if cards[revealed_cards[0][0]] == cards[revealed_cards[1][0]]:
                    matches += 1
                    revealed_cards.clear()
                    if matches == len(images):
                        game_result_label.config(text=f"Congratulations! You completed the game in {moves} moves.")
                        pygame.mixer.music.stop()
                else:
                    game_root.after(1000, hide_cards)
    def hide_cards():
        for idx, btn in revealed_cards:
            btn.config(image=hidden_image)
        revealed_cards.clear()
    def exit_game():
        pygame.mixer.music.stop()
        game_root.destroy()
    for i in range(4):
        for j in range(3):
            index = i * 3 + j
            if index < len(cards):
                button = tk.Button(game_root, image=hidden_image)
                button.grid(row=i, column=j, padx=5, pady=5)
                button.config(command=lambda idx=index, btn=button: on_card_click(idx, btn))
    exit_button = tk.Button(game_root, text="Exit", command=exit_game, font=("Maiandra GD", 14))
    exit_button.grid(row=6, column=0, columnspan=4, pady=10)
    game_root.mainloop()

def play_meditation_music(meditation_window, music_file, gif_file, gif_size=(400, 400)):  
    # Set window size (Fix the geometry issue)
    meditation_window.geometry("650x600")  
    meditation_window.configure(bg="light blue")

    # Initialize text-to-speech engine
    tts_engine = pyttsx3.init()
    tts_engine.say("Welcome to your meditation session. Relax, take a deep breath, and let go of stress.")
    tts_engine.runAndWait()

    # Start background music in a separate thread to prevent blocking
    def play_music():
        pygame.mixer.init()
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play(-1)  # Loop indefinitely

    threading.Thread(target=play_music, daemon=True).start()  # Run music in background

    # Meditation text label
    label = tk.Label(meditation_window, text="Relax and Enjoy the Meditation", font=("Maiandra GD", 16), fg="blue", bg="light blue")
    label.pack(pady=20)

    # Create label for GIF animation
    gif_label = tk.Label(meditation_window, bg="light blue")
    gif_label.pack(pady=10)

    # Load and resize the animated GIF
    pil_img = Image.open(gif_file)
    frames = [ImageTk.PhotoImage(frame.copy().convert("RGBA").resize(gif_size, Image.LANCZOS)) 
              for frame in ImageSequence.Iterator(pil_img)]  # Resize GIF frames

    # Function to update GIF animation
    def update_gif(ind):
        gif_label.configure(image=frames[ind])
        meditation_window.after(100, update_gif, (ind + 1) % len(frames))

    # Start GIF animation
    update_gif(0)

    # Stop Meditation Button
    stop_button = tk.Button(meditation_window, text="Stop Meditation", command=lambda: stop_meditation(meditation_window), font=("Arial", 14))
    stop_button.pack(pady=10)

def stop_meditation(meditation_window):
    pygame.mixer.music.stop()
    meditation_window.destroy()
# --------------------
# Login/Sign Up Pages
# --------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_page():
    global login_root, username_entry, password_entry
    login_root = tk.Tk()
    login_root.title("Login Page")
    login_root.geometry("700x600")
    login_root.configure(bg="light blue")

    try:
        pil_image = Image.open("./images/logo1.webp")
        try:
            resample_filter = Image.Resampling.LANCZOS  # For Pillow >= 10.0
        except AttributeError:
            resample_filter = Image.LANCZOS  # For older Pillow versions
        
        resized_image = pil_image.resize((100, 100), resample=resample_filter)
        logo = ImageTk.PhotoImage(resized_image)
        logo_label = tk.Label(login_root, image=logo, bg="light blue")
        logo_label.image = logo  # Keep a reference
        logo_label.pack(pady=10)
    except Exception as e:
        print(f"Error loading image: {e}")

    tk.Label(login_root, text="Login", font=("Maiandra GD", 18, "bold"), bg="light blue").pack(pady=10)

    # Create a frame for alignment using grid
    form_frame = tk.Frame(login_root, bg="light blue")
    form_frame.pack(pady=10)

    # Email Label and Entry (Proper Alignment with Grid)
    tk.Label(form_frame, text="Email:", font=("Maiandra GD", 12), bg="light blue", width=12, anchor="e").grid(row=0, column=0, padx=5, pady=5)
    username_entry = tk.Entry(form_frame, width=30)
    username_entry.grid(row=0, column=1, padx=5, pady=5)

    # Password Label and Entry (Proper Alignment with Grid)
    tk.Label(form_frame, text="Password:", font=("Maiandra GD", 12), bg="light blue", width=12, anchor="e").grid(row=1, column=0, padx=5, pady=5)
    password_entry = tk.Entry(form_frame, width=30, show="*")
    password_entry.grid(row=1, column=1, padx=5, pady=5)

    # Frame for Buttons
    button_frame = tk.Frame(login_root, bg="light blue")
    button_frame.pack(pady=10, anchor="center")  
    login_button = tk.Button(button_frame, text="Login", font=("Maiandra GD", 12), command=login_action)
    login_button.pack(side="left", padx=10)
    signup_button = tk.Button(button_frame, text="Sign Up", font=("Maiandra GD", 12), command=signup_page)
    signup_button.pack(side="left", padx=10)

    login_root.mainloop()
def login_action():
    global user_name, user_id, user_email, login_root
    email = username_entry.get()
    password = password_entry.get()
    result = check_user_exists(email)
    if result:
        user_id, stored_name = result
        user_name = stored_name
        user_email = email
        # Password check could be added here (e.g. comparing hashed passwords)
        tts_engine.say(f"Welcome back, {user_name}!")
        tts_engine.runAndWait()
        messagebox.showinfo("Login Successful", f"Welcome back, {user_name}!")
        login_root.destroy()
        start_gui()
    else:
        tts_engine.say("Login Failed: User not found. Please sign up.")
        tts_engine.runAndWait()
        messagebox.showerror("Login Failed", "User not found. Please sign up.")
def signup_page():
    global signup_root, signup_username_entry, signup_password_entry, login_root
    login_root.destroy()
    
    signup_root = tk.Tk()
    signup_root.title("Sign Up")
    signup_root.geometry("700x600")
    signup_root.configure(bg="light blue")
    
    try:
        pil_image = Image.open("./images/logo1.webp")
        try:
            resample_filter = Image.Resampling.LANCZOS  # For Pillow >= 10.0
        except AttributeError:
            resample_filter = Image.LANCZOS  # For older Pillow versions

        resized_image = pil_image.resize((100, 100), resample=resample_filter)
        logo = ImageTk.PhotoImage(resized_image)
        logo_label = tk.Label(signup_root, image=logo, bg="light blue")
        logo_label.image = logo  # Keep a reference
        logo_label.pack(pady=10)
    except Exception as e:
        print(f"Error loading image: {e}")

    tk.Label(signup_root, text="Sign Up", font=("Maiandra GD", 18, "bold"), bg="light blue").pack(pady=20)

    # Create a frame for proper alignment using grid
    form_frame = tk.Frame(signup_root, bg="light blue")
    form_frame.pack(pady=10)

    # Email Label and Entry (Proper Alignment with Grid)
    tk.Label(form_frame, text="Email:", font=("Maiandra GD", 12), bg="light blue", width=12, anchor="e").grid(row=0, column=0, padx=5, pady=5)
    signup_username_entry = tk.Entry(form_frame, width=30)
    signup_username_entry.grid(row=0, column=1, padx=5, pady=5)

    # Password Label and Entry (Proper Alignment with Grid)
    tk.Label(form_frame, text="Password:", font=("Maiandra GD", 12), bg="light blue", width=12, anchor="e").grid(row=1, column=0, padx=5, pady=5)
    signup_password_entry = tk.Entry(form_frame, width=30, show="*")
    signup_password_entry.grid(row=1, column=1, padx=5, pady=5)

    # Create Account Button (Centered)
    tk.Button(signup_root, text="Create Account", font=("Maiandra GD", 12), command=register_user).pack(pady=20)

    signup_root.mainloop()


def register_user():
    global user_name, user_id, signup_root, user_email
    email = signup_username_entry.get()
    password = signup_password_entry.get()
    if not email or not password:
        tts_engine.say("Sign Up Failed: Email and Password cannot be empty!")
        tts_engine.runAndWait()
        messagebox.showwarning("Sign Up Failed", "Email and Password cannot be empty!")
        return
    if check_user_exists(email):
        messagebox.showerror("Sign Up Failed", "Email already exists!")
        return
    tts_engine.say("Sign Up: Please provide additional details via speech input.")
    tts_engine.runAndWait()
    messagebox.showinfo("Sign Up", "Please provide additional details via speech input.")
    
    def get_audio_input(prompt):
        print(prompt)
        tts_engine.say(prompt)
        tts_engine.runAndWait()
        audio_data, sample_rate = record_audio(duration=5)
        audio_file = save_audio_to_wav(audio_data, sample_rate, filename="temp_audio_input.wav")
        transcription = transcribe_audio(audio_file)
        if not transcription:
            tts_engine.say("Could not capture your input. Please try again.")
            tts_engine.runAndWait()
            return get_audio_input(prompt)
        return transcription
    
    spoken_name = get_audio_input("Please say your name.")
    age = get_audio_input("Please say your age.")
    gender = get_audio_input("Please say your gender.")
    address = get_audio_input("Please say your address.")
    user_id = store_user_details(spoken_name, age, gender, email, address)
    user_name = spoken_name
    user_email = email
    messagebox.showinfo("Sign Up Successful", f"Account created! Welcome, {user_name}.")
    signup_root.destroy()
    start_gui()

def login_action():
    global user_name, user_id, user_email, login_root
    email = username_entry.get()
    password = password_entry.get()
    result = check_user_exists(email)
    if result:
        user_id, stored_name = result
        user_name = stored_name
        user_email = email
        # Password check could be added here (e.g. comparing hashed passwords)
        tts_engine.say(f"Welcome back, {user_name}!")
        tts_engine.runAndWait()
        messagebox.showinfo("Login Successful", f"Welcome back, {user_name}!")
        login_root.destroy()
        start_gui()
    else:
        tts_engine.say("Login Failed: User not found. Please sign up.")
        tts_engine.runAndWait()
        messagebox.showerror("Login Failed", "User not found. Please sign up.")
def signup_page():
    global signup_root, signup_username_entry, signup_password_entry, login_root
    login_root.destroy()
    
    signup_root = tk.Tk()
    signup_root.title("Sign Up")
    signup_root.geometry("700x600")
    signup_root.configure(bg="light blue")
    
    try:
        pil_image = Image.open("./images/logo1.webp")
        try:
            resample_filter = Image.Resampling.LANCZOS  # For Pillow >= 10.0
        except AttributeError:
            resample_filter = Image.LANCZOS  # For older Pillow versions

        resized_image = pil_image.resize((100, 100), resample=resample_filter)
        logo = ImageTk.PhotoImage(resized_image)
        logo_label = tk.Label(signup_root, image=logo, bg="light blue")
        logo_label.image = logo  # Keep a reference
        logo_label.pack(pady=10)
    except Exception as e:
        print(f"Error loading image: {e}")

    tk.Label(signup_root, text="Sign Up", font=("Maiandra GD", 18, "bold"), bg="light blue").pack(pady=20)

    # Create a frame for proper alignment using grid
    form_frame = tk.Frame(signup_root, bg="light blue")
    form_frame.pack(pady=10)

    # Email Label and Entry (Proper Alignment with Grid)
    tk.Label(form_frame, text="Email:", font=("Maiandra GD", 12), bg="light blue", width=12, anchor="e").grid(row=0, column=0, padx=5, pady=5)
    signup_username_entry = tk.Entry(form_frame, width=30)
    signup_username_entry.grid(row=0, column=1, padx=5, pady=5)

    # Password Label and Entry (Proper Alignment with Grid)
    tk.Label(form_frame, text="Password:", font=("Maiandra GD", 12), bg="light blue", width=12, anchor="e").grid(row=1, column=0, padx=5, pady=5)
    signup_password_entry = tk.Entry(form_frame, width=30, show="*")
    signup_password_entry.grid(row=1, column=1, padx=5, pady=5)

    # Create Account Button (Centered)
    tk.Button(signup_root, text="Create Account", font=("Maiandra GD", 12), command=register_user).pack(pady=20)

    signup_root.mainloop()


def register_user():
    global user_name, user_id, signup_root, user_email
    email = signup_username_entry.get()
    password = signup_password_entry.get()
    if not email or not password:
        tts_engine.say("Sign Up Failed: Email and Password cannot be empty!")
        tts_engine.runAndWait()
        messagebox.showwarning("Sign Up Failed", "Email and Password cannot be empty!")
        return
    if check_user_exists(email):
        messagebox.showerror("Sign Up Failed", "Email already exists!")
        return
    tts_engine.say("Sign Up: Please provide additional details via speech input.")
    tts_engine.runAndWait()
    messagebox.showinfo("Sign Up", "Please provide additional details via speech input.")
    
    def get_audio_input(prompt):
        print(prompt)
        tts_engine.say(prompt)
        tts_engine.runAndWait()
        audio_data, sample_rate = record_audio(duration=5)
        audio_file = save_audio_to_wav(audio_data, sample_rate, filename="temp_audio_input.wav")
        transcription = transcribe_audio(audio_file)
        if not transcription:
            tts_engine.say("Could not capture your input. Please try again.")
            tts_engine.runAndWait()
            return get_audio_input(prompt)
        return transcription
    
    spoken_name = get_audio_input("Please say your name.")
    age = get_audio_input("Please say your age.")
    gender = get_audio_input("Please say your gender.")
    address = get_audio_input("Please say your address.")
    user_id = store_user_details(spoken_name, age, gender, email, address)
    user_name = spoken_name
    user_email = email
    messagebox.showinfo("Sign Up Successful", f"Account created! Welcome, {user_name}.")
    signup_root.destroy()
    start_gui()
# --------------------
# Main GUI Setup
# --------------------
def start_gui():
    global result_label, root, gif_label, user_name
    root = tk.Tk()
    root.title("Sentiowave")
    root.geometry("900x700")
    root.configure(bg="light blue")
    
    greeting_text = f"Hello, {user_name}! " if user_name else ""
    result_label = tk.Label(root, text=greeting_text + "Please provide your details.", 
                            font=("Maiandra GD", 19), bg="light blue")
    result_label.pack(pady=20)
    
    # Create a label to hold the animated GIF
    gif_label = tk.Label(root, bg="light blue")
    gif_label.pack(pady=10)
    
    try:
        pil_img = Image.open("./images/confused.gif")
        frames = [ImageTk.PhotoImage(frame.copy().convert("RGBA"))
                  for frame in ImageSequence.Iterator(pil_img)]
        
        def update(ind):
            # Check if gif_label still exists
            if not gif_label.winfo_exists():
                return  # Widget is gone, so stop the update loop
            
            frame = frames[ind]
            gif_label.config(image=frame)
            gif_label.image = frame  # Keep a reference to prevent garbage collection
            # Schedule next frame update (loop indefinitely)
            root.after(100, update, (ind + 1) % len(frames))
        
        update(0)
    except Exception as e:
        print("Error loading animated GIF:", e)
    
    # Function to handle speaking asynchronously, using user_name
    def start_speech():
       greeting_message = f"Hello {user_name},I am Rajesh, welcome to Sentiowave! I am excited to assist you with your analysis today. Let’s get started!"
       tts_engine.say(greeting_message)
       tts_engine.runAndWait()

    
    # Start speech in a separate thread
    threading.Thread(target=start_speech, daemon=True).start()
    
    start_button = tk.Button(root, text="Start Analysis", font=("Maiandra GD", 14),
                             command=ask_for_details)
    start_button.pack(pady=20)
    
    root.mainloop()

# --------------------
# Start the Application with Login Page
# --------------------
if __name__ == "__main__":
    login_page()
