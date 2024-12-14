import cv2
import torch
from torchvision import transforms
from PIL import Image
from models.emotion_model import EmotionRecognitionCNN

def detect_emotions():
    categories = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = EmotionRecognitionCNN(num_classes=len(categories)).to(device)
    model.load_state_dict(torch.load("../checkpoints/emotion_recognition_model.pth", map_location=device))
    model.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Webcam initialization
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Image.fromarray(face)
            face = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(face)
                _, predicted = torch.max(outputs, 1)
                emotion = categories[predicted.item()]

            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions()
