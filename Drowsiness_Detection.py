import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame

# Initialize pygame mixer
pygame.mixer.init()


sender_email = "" # type the sender email 
receiver_email = "" # type the reciver mail
password = "" # put your app password - for reference check google accounts manager

def send_email():
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Emergency: Alert!"

    body = "The driver might be in danger."
    msg.attach(MIMEText(body, 'plain'))

    # Connect to the SMTP server
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()  # Secure the connection
        # Login with your credentials
        server.login(sender_email, password)
        
        server.sendmail(sender_email, receiver_email, msg.as_string())

def play_alert_sound():
    alert_sound = pygame.mixer.Sound('siren-alert-96052.mp3')
    alert_sound.play()
    return alert_sound

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check_sound = 20
frame_check_email = 150
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag_sound = 0
flag_email = 0
sound_playing = False
eyes_detected = False

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        eyes_detected = True  # Eyes are detected
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < thresh:
            flag_sound += 1
            flag_email += 1
            if flag_sound >= frame_check_sound and not sound_playing:
                alert_sound = play_alert_sound()  # Play alert sound
                sound_playing = True
            if flag_email >= frame_check_email:
                send_email()  # Send email to emergency contact
                flag_email = 0
        else:
            flag_sound = 0
            flag_email = 0
            if sound_playing:
                alert_sound.stop()
                sound_playing = False
    else:

        if not eyes_detected:
            cv2.putText(frame, "No eyes detected. Please position your face in the camera view.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Reset the eyes_detected flag for the next iteration
    eyes_detected = False

    # Flip the frame horizontally before displaying
    frame = cv2.flip(frame, 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
