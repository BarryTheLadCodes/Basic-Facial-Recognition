face_detection.py is a simple python script for detecting human faces in an image using OpenCV and Haar cascades.
faces_train.py uses this face detection system to find faces in labeled images with the name the face should be associated to, and then trains on the image set. Facial recogniztion data is then outputed to a .yml file for use elsewhere.
face_recognition.py first uses face detection to look for faces in the given image, then uses the .yml file to attempt to recognize them.
