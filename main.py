import cv2
import numpy as np
import torch
from fer import FER
from speechbrain.pretrained import EncoderClassifier
import sounddevice as sd
import librosa

face_detector = FER(mtcnn=True)  # facial emotion
speech_model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion_recognition"
)

def analyze_face(frame):
    results = face_detector.detect_emotions(frame)
    if results:
        emotions = results[0]["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        return top_emotion, emotions[top_emotion]
    return "neutral", 0.0

def analyze_voice(duration=3, fs=16000):
    print("🎙 Recording voice sample...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    wav = np.squeeze(audio)
    wav = torch.tensor(wav).unsqueeze(0)
    prediction = speech_model.classify_batch(wav)
    label = prediction[3][0]  # label string
    score = float(torch.max(torch.softmax(prediction[1], dim=1)))
    return label, score

def demo_loop():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze face
        emotion, score = analyze_face(frame)
        cv2.putText(frame, f"Face Emotion: {emotion} ({score:.2f})", 
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("AI Lie Detector Demo", frame)

        # Keyboard interaction
        key = cv2.waitKey(1) & 0xFF
        if key == ord('v'):  # record voice
            label, vscore = analyze_voice()
            print(f"Voice Emotion: {label} ({vscore:.2f})")
            # naive fusion = random mix
            deception_score = 1 - ((score + vscore) / 2)
            print(f"🤖 Demo 'Deception Score': {deception_score:.2f}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo_loop()True
