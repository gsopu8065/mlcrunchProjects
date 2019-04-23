import cv2

faceCascadeDetector = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")

#video source from webcam
camera = cv2.VideoCapture(0)

while True:
    # read the current frame
    (grabbed, frame) = camera.read()

    # check frame is fetched or not
    if not grabbed:
        break

    #resixe the frame
    frame = cv2.resize(frame, (700, 500))

    #convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces using haar cascade approach
    faces = faceCascadeDetector.\
        detectMultiScale(gray,scaleFactor=1.1,
        minNeighbors=5, minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # draw faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y),
            (x + w, y + h), (0, 255, 0), 2)

    # show the frame till u enter space
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the space key is pressed, stop the loop
    if key == ord(" "):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()