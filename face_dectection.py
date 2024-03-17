import cv2 as cv

capture = cv.VideoCapture(0)

def rescaleFrame(frame, scale=1.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame)
    cv.imshow("Camera", frame_resized)

    gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)

    haar_cascade_face = cv.CascadeClassifier('haar_face.xml')

    faces_rect = haar_cascade_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)

    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame_resized, (x,y), (x+w,y+h), (0,0,255), 3)

    #haar_cascade_lower_body = cv.CascadeClassifier('haar_lower_body.xml')

    #lower_body_rect = haar_cascade_lower_body.detectMultiScale(gray, 1.1, 4)

    #for (x,y,w,h) in lower_body_rect:
        #cv.rectangle(frame_resized, (x,y), (x+w,y+h), (0,0,255), 3)

    cv.imshow("Face Detector", frame_resized)

    if cv.waitKey(20) & 0xFF==ord(' '):
        break

capture.release()
cv.destroyAllWindows()