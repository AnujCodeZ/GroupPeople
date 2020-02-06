import face_recognition
import numpy as np 
import os

filepath = "Data"

face_database = {}
for name in os.listdir(filepath):
    identity = os.path.splitext(os.path.basename(name))[0]
    img = face_recognition.load_image_file(os.path.join(filepath,name))
    face_database[identity] = face_recognition.face_encodings(img)

np.save("database.npy",face_database)

for name in os.listdir(filepath):
    Identity = os.path.splitext(os.path.basename(name))[0]
    try:
        os.mkdir("Groups/"+Identity)
    except:
        print("Filename "+Identity+" Exists")