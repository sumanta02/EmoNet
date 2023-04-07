from flask import Flask, render_template, Response, stream_with_context, request, redirect, url_for
import cv2
import os
import torch
from torchvision.transforms import transforms
emotion = "No Face detected"
face_classifier = cv2.CascadeClassifier('harrcascade_frontallface_default.xml')
classifier = torch.load('model_ft.h5',map_location ='cpu')
classifier.eval()

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = Flask(__name__,  static_folder="static")

def process_image(img_path):
    emotion="No Face Detected"
    print(img_path)
    frame = cv2.imread(img_path)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    # Convert the frame to a PyTorch tensor
    img_tensor = transform(frame)

    # Add a batch dimension to the tensor
    img_tensor = img_tensor.unsqueeze(0)
    face_detected = False
    for (x,y,w,h) in faces:
        face_detected = True
        with torch.no_grad():
            prediction = classifier(img_tensor)
            label=emotion_labels[prediction.argmax()]
    if face_detected:
        emotion = label
        if label in ['Angry','Disgust','Fear','Sad']:
            emotion = label + " emotion detected. Please Seek Help. Speak with mental health experts."
        elif label in ['Happy','Neutral', 'Surprise']:
            emotion = label + " emotion detected."
        if label not in emotion_labels:
            emotion = "No Face detected"
    return(emotion)
    


def generate_frames():
    camera = cv2.VideoCapture(0)  # 0 -> index of camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            labels = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            # Convert the frame to a PyTorch tensor
            img_tensor = transform(frame)

            # Add a batch dimension to the tensor
            img_tensor = img_tensor.unsqueeze(0)
            face_detected = False
            for (x,y,w,h) in faces:
                face_detected = True
                with torch.no_grad():
                    prediction = classifier(img_tensor)
                    label=emotion_labels[prediction.argmax()]
            if face_detected:
                global emotion
                emotion = label
                if label in ['Angry','Disgust','Fear','Sad']:
                    emotion = label + " emotion detected. Please Seek Help. Speak with mental health experts."
                elif label in ['Happy','Neutral', 'Surprise']:
                    emotion = label + " emotion detected."
                if label not in emotion_labels:
                    emotion = "No Face detected"
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

@app.route('/')
def index():
    return render_template('VideoProcessing.html')

@app.route('/image_processing')
def image_processing():
    return render_template('imageProcessing.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            file.save("static/" + os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imgPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            emotion = process_image("static/" + imgPath)
            
    return render_template('result.html', image_path=url_for("static", filename=imgPath), text=emotion)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/text_feed')
def text_feed():
    while True:
        return Response(gen_text(), mimetype='text/plain')

def gen_text():
    while True:
        return emotion
if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'

    app.run(debug=True)
