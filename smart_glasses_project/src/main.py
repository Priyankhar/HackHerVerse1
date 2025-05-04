import threading
import torch
import cv2
import time
import os
from dotenv import load_dotenv
import openai
from playsound import playsound
import pyttsx3
import tempfile
from gtts import gTTS
import speech_recognition as sr

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OpenAI API key not loaded. Check your .env file.")
openai.api_key = api_key

# Load YOLOv5 model
print("⏳ Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', source='github')
print("✅ YOLOv5 loaded successfully!")

# pyttsx3 for English
engine = pyttsx3.init()
engine.setProperty('rate', 150)

last_sentence_spoken = ""
last_gpt_description = ""
last_object_sentence = ""
language = "en"

def speak(text, lang="en"):
    global last_sentence_spoken
    last_sentence_spoken = text
    if lang == "en":
        engine.say(text)
        engine.runAndWait()
    elif lang == "ta":
        try:
            tts = gTTS(text=text, lang='ta')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_path = fp.name
            tts.save(temp_path)
            playsound(temp_path)
            os.remove(temp_path)
        except Exception as e:
            print(f"⚠️ Tamil TTS error: {e}")

def manual_language_fallback():
    while True:
        choice = input("⌨️  Press 'T' for Tamil or 'E' for English: ").strip().lower()
        if choice == 't':
            return "ta"
        elif choice == 'e':
            return "en"
        else:
            print("❌ Invalid input. Please enter T or E.")

def select_language_by_voice():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    speak("Please say your language. Say English or Tamil.", lang="en")
    print("🎤 Listening for language selection...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            response = recognizer.recognize_google(audio).lower().strip()
            print(f"✅ You said: {response}")

            tamil_keywords = ["tamil", "tamizh", "tamir", "தமிழ்"]
            english_keywords = ["english", "inglish"]

            if any(word in response for word in tamil_keywords):
                return "ta"
            elif any(word in response for word in english_keywords):
                return "en"
            else:
                print("🤷 Unrecognized language. Falling back to keyboard input.")
                return manual_language_fallback()
        except Exception as e:
            print(f"⚠️ Voice input error: {e}")
            speak("I did not hear you. Please press T for Tamil or E for English.", lang="en")
            return manual_language_fallback()

language = select_language_by_voice()

translations_ta = {
    "person": "நபர்",
    "bicycle": "மிதிவண்டி",
    "car": "கார்",
    "motorcycle": "மோட்டார்சைக்கிள்",
    "bus": "பேருந்து",
    "truck": "லாரி",
    "traffic light": "போக்குவரத்து விளக்கு",
    "stop sign": "நிறுத்தக் குறி",
    "bench": "நாற்காலி",
    "bird": "பறவை",
    "cat": "பூனை",
    "dog": "நாய்",
    "backpack": "பையின்",
    "umbrella": "குடை"
}
translations_en = {k: k for k in translations_ta}

def translate_object(name):
    return translations_ta.get(name, name) if language == "ta" else name

def get_direction_from_bbox(x1, x2, frame_width):
    x_center = (x1 + x2) / 2
    if x_center < frame_width / 3:
        return "to your left"
    elif x_center > 2 * frame_width / 3:
        return "to your right"
    else:
        return "ahead"

def handle_command(command):
    global language, last_gpt_description, last_object_sentence
    if "describe again" in command and last_gpt_description:
        print("🔁 Repeating GPT description...")
        speak(last_gpt_description, lang=language)
    elif "repeat last object" in command and last_object_sentence:
        print("🔁 Repeating last object description...")
        speak(last_object_sentence, lang=language)
    elif "switch to tamil" in command:
        language = "ta"
        speak("மொழி தமிழ் ஆக மாற்றப்பட்டது", lang="ta")
    elif "switch to english" in command:
        language = "en"
        speak("Language switched to English.", lang="en")

def listen_for_commands(command_callback):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        try:
            with mic as source:
                print("🎤 Voice command listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower().strip()
                print(f"🎙️ Heard: {command}")
                command_callback(command)
        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except Exception as e:
            print(f"⚠️ Voice command error: {e}")

threading.Thread(target=listen_for_commands, args=(handle_command,), daemon=True).start()

video_files = ["videos/obs1.mp4", "videos/vid1.mp4"]

for video_path in video_files:
    print(f"\n🎞️ Opening video: {video_path}")
    if not os.path.isfile(video_path):
        print(f"❌ File not found: {video_path}. Skipping...")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        abs_path = os.path.abspath(video_path)
        print(f"🔄 Trying absolute path: {abs_path}")
        cap = cv2.VideoCapture(abs_path)

    if not cap.isOpened():
        print(f"❌ Failed to open: {video_path}. Skipping...")
        continue

    print(f"✅ Opened {video_path} successfully!")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("⚠️ Invalid FPS. Defaulting to 30.")
        fps = 30

    last_announcement_time = 0
    last_gpt_time = 0
    announcement_interval = 8
    gpt_interval = 15
    recent_objects = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            cv2.imshow("Smart Glasses", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frame_width = frame.shape[1]
        results = model(frame)
        detections = results.xyxy[0]
        names = results.names

        new_object_descriptions = []

        for det in detections:
            x1, y1, x2, y2, conf, class_id = det.tolist()
            class_id = int(class_id)
            label = names[class_id]
            direction = get_direction_from_bbox(x1, x2, frame_width)

            key = (label, direction)
            if key not in recent_objects:
                translated = translate_object(label)
                if language == "ta":
                    if direction == "ahead":
                        desc = f"எனக்கு முன் ஒரு {translated}"
                    elif direction == "to your left":
                        desc = f"எனது இடப்புறம் ஒரு {translated}"
                    else:
                        desc = f"எனது வலப்புறம் ஒரு {translated}"
                else:
                    desc = f"A {translated} {direction}"
                new_object_descriptions.append(desc)
                recent_objects.add(key)

        current_time = time.time()

        if new_object_descriptions and (current_time - last_announcement_time > announcement_interval):
            if language == "ta":
                sentence = " , ".join(new_object_descriptions) + " இருக்கிறது."
            else:
                sentence = "I see " + ", ".join(new_object_descriptions) + "."

            print(f"🗣️ {sentence}")
            speak(sentence, lang=language)
            last_object_sentence = sentence
            time.sleep(1.5)
            last_announcement_time = current_time

        if new_object_descriptions and (current_time - last_gpt_time > gpt_interval):
            gpt_prompt = f"A person is walking and sees: {', '.join(new_object_descriptions)}. Describe the scene for a visually impaired person."
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant narrating scenes to visually impaired people."},
                        {"role": "user", "content": gpt_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
                description = response['choices'][0]['message']['content']
                print(f"🧠 GPT says: {description}")
                speak(description, lang=language)
                last_gpt_description = description
                time.sleep(1.5)
            except Exception as e:
                print(f"⚠️ GPT error: {e}")
                time.sleep(1.5)

            last_gpt_time = current_time

        annotated = results.render()[0]
        cv2.imshow("Smart Glasses", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()
print("\n✅ All videos processed.")
