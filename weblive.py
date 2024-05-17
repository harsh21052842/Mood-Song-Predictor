import os
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import tkinter as tk
from PIL import Image, ImageTk

model = load_model("feelings_model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

class EmotionProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        try:
            self.emotion = np.load("emotion.npy")[0]
        except:
            self.emotion = ""

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frm = cv2.flip(frame, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1,-1)
            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)
            self.emotion = pred
            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
                                connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        cv2.imshow("Emotion Detection", frm)  # Show the processed frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()

        return frm

emotion_processor = EmotionProcessor()

def recommend_music():
    lang = language_entry.get()
    singer = singer_entry.get()      

    if lang and singer:
        frame = emotion_processor.process_frame()  # Capture emotion
        if not emotion_processor.emotion:
            result_label.config(text="Please let me capture your emotion first")
        else:
            webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion_processor.emotion}+song+{singer}")
            np.save("emotion.npy", np.array([""]))

# Create window
root = tk.Tk()
root.title("Music Recommendation System")
root.geometry('800x600')
root.configure(bg='black')  # Set background color of the window to black

# Function to create a label with an image and text below it
def create_image_label(image_path, text, x, y):
    image = Image.open(image_path).resize((64, 64))
    tk_image = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=tk_image, bg='black')  # Set background color of the image label to black
    image_label.image = tk_image
    image_label.place(x=x, y=y)

    text_label = tk.Label(root, text=text, bg='black', fg='white')  # Set background color of the text label to black and foreground color to white
    text_label.place(x=x, y=y + 70)

# Add images to the root window and display their names below them
image_paths = [
    ('smile.png', 'Smile'),
    ('sad.png', 'Sad'),
    ('angry.png', 'Angry'),
    ('gym.png', 'Gym'),
    ('party.png', 'Party'),
    ('Romantic.png', 'Romantic')
]

total_width = len(image_paths) * 90
spacing = (800 - total_width) / 2

for i, (image_path, text) in enumerate(image_paths):
    create_image_label(os.path.join(os.path.dirname(__file__), image_path), text, spacing + i * 90, 250)

# Add Emotion Music Suggestor Label
emotion_label = tk.Label(root, text="Emotion Detection Music Recommendation System", bg='black', fg='white', font=("Arial", 16, "bold"))
emotion_label.pack(pady=(20, 0))  # Add padding above the label and no padding below

# Center align Language Label and Entry
language_frame = tk.Frame(root, bg='black')  # Set background color of the frame to black
language_frame.pack(pady=(10, 0))  # Add padding above Language frame and no padding below
language_label = tk.Label(language_frame, text="Language:", bg='black', fg='white')  # Set background color of the label to black and foreground color to white
language_label.pack(side="left")
language_entry = tk.Entry(language_frame, bg='black', fg='white')  # Set background color of the entry widget to black and foreground color to white
language_entry.pack(side="left")

# Center align Singer Label and Entry
singer_frame = tk.Frame(root, bg='black')  # Set background color of the frame to black
singer_frame.pack(pady=10)  # Add padding above Singer frame
singer_label = tk.Label(singer_frame, text="Singer:", bg='black', fg='white')  # Set background color of the label to black and foreground color to white
singer_label.pack(side="left")
singer_entry = tk.Entry(singer_frame, bg='black', fg='white')  # Set background color of the entry widget to black and foreground color to white
singer_entry.pack(side="left")


# Recommend Music Button
recommend_button = tk.Button(root, text="Recommend Music", command=recommend_music, bg='black', fg='white')  # Set background color of the button to black and foreground color to white
recommend_button.pack(pady=10)

# Add label indicating the system can detect emotions
emotion_detect_label = tk.Label(root, text="Following emotions can be detected by this model :", bg='black', fg='white', font=("Arial", 12))
emotion_detect_label.pack(pady=(20, 10))  # Add padding above the label and below


# Result Label
result_label = tk.Label(root, text="", bg='black', fg='white')  # Set background color of the label to black and foreground color to white
result_label.pack(pady=10)

root.mainloop()
