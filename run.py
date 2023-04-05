from flask import Flask, render_template, Response, stream_with_context
import cv2
import torch
from torchvision.transforms import transforms

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

app = Flask(__name__)

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
            for (x,y,w,h) in faces:
                with torch.no_grad():
                    prediction = classifier(img_tensor)
                    label=emotion_labels[prediction.argmax()]
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            font_scale = 1
            color = (0, 255, 0)
            thickness = 2
            if label in ['Angry','Disgust','Fear', 'Sad']:
                color = (0, 0, 255)
            cv2.putText(frame, label, org, font, font_scale, color, thickness, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, render_template, Response
import cv2
import torch
from torchvision import transforms

"""face_classifier = cv2.CascadeClassifier('harrcascade_frontallface_default.xml')
classifier = torch.load('model_ft.h5', map_location='cpu')
classifier.eval()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = Flask(__name__)

def generate_frames():
    camera = cv2.VideoCapture(0)  # 0 -> index of camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                # Crop the face region
                face = frame[y:y+h, x:x+w]

                # Convert the face region to a PyTorch tensor
                img_tensor = transform(face)

                # Add a batch dimension to the tensor
                img_tensor = img_tensor.unsqueeze(0)

                # Pass the tensor through the classifier
                with torch.no_grad():
                    prediction = classifier(img_tensor)
                    label = emotion_labels[prediction.argmax()]
                    labels.append(label)

                # Draw the label on the frame
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Join the labels into a single string and display it on the frame
            label_str = ', '.join(labels)
            cv2.putText(frame, label_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
 """

""" from flask import Flask, render_template, Response

face_classifier = cv2.CascadeClassifier('harrcascade_frontallface_default.xml')
classifier = torch.load('model_ft.h5', map_location='cpu')
classifier.eval()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    camera = cv2.VideoCapture(0)  # 0 -> index of camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            # Convert the frame to a PyTorch tensor
            img_tensor = transform(frame)

            # Add a batch dimension to the tensor
            img_tensor = img_tensor.unsqueeze(0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #text = 'Hello, wdddorld!'
            org = (50, 50)
            font_scale = 1
            color = (0, 255, 0)
            thickness = 2
            #cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
            for (x, y, w, h) in faces:
                with torch.no_grad():
                    prediction = classifier(img_tensor)
                    label = emotion_labels[prediction.argmax()]
                    labels.append(label)
            # Draw bounding boxes around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the current emotion label in the header
            if labels:
                emotion_label = max(set(labels), key=labels.count)
                header = f"Current emotion: {emotion_label}"
            else:
                header = "No faces detected"

            # Combine the frame and the header into a single image
            #cv2.putText(frame, header, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
 """

""" import cv2
from flask import Flask, Response

app = Flask(__name__)

def gen_frames():
    Generator function that captures video frames from the webcam and adds a text overlay.
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Draw text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'Hello, wdddorld!'
        org = (50, 50)
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

        # Convert the frame to a JPEG image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in a Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the webcam and cleanup
    cap.release()

@app.route('/')
def index():
    Route that returns a streaming response of video frames from the webcam with a text overlay.
    return Response(write_text(gen_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

def write_text(frames):
    Generator function that writes text in the response.
    for frame in frames:
        yield frame
        yield b'Hello, world!\n'

if __name__ == '__main__':
    app.run(debug=True) """