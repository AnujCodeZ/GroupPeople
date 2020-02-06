import os
import numpy as np
import face_recognition
from shutil import copyfile

filepath = "New"

src = "/home/anujrana/Documents/Code/AI/GroupPeople/New/"
dest = "/home/anujrana/Documents/Code/AI/GroupPeople/Groups/"

database = np.load("database.npy").item()

known_face_encodings = list()
known_face_names = list()

for key, value in database.items():
    known_face_encodings.append(value[0])
    known_face_names.append(key)

for fname in os.listdir(filepath):
    img = face_recognition.load_image_file(os.path.join(filepath,fname))
    encodings = face_recognition.face_encodings(img)[0]

    matches = face_recognition.compare_faces(known_face_encodings, encodings)

    name = None
    face_distances = face_recognition.face_distance(known_face_encodings, encodings)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    
    copyfile(src+fname,dest+name+"/"+fname)
    
    

