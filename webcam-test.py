import cv2
import face_recognition as recog
from PIL import Image, ImageDraw

cam = cv2.VideoCapture(0)

pariImage = recog.load_image_file("pari.jpg")
pariEnc = recog.face_encodings(pariImage)

known_faces = []
known_faces.append(pariEnc)
known_names = ["pari"]


while True:
	ret, img = cam.read()
	img = cv2.flip(img, 1)

	face_locations = recog.face_locations(img)
	face_encodings = recog.face_encodings(img, face_locations)

	for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
		
		matches = recog.compare_faces(known_faces, face_encoding)

		name = "unknown person"
		
		#print("Matches: "+str(matches))

		for x in matches:
			index = matches.index(x)
			name = known_names[index]

		cv2.rectangle(img, (left, top), (right, bottom), (255,0,0), 2)
		cv2.putText(img,str(name),(left, top),cv2.FONT_HERSHEY_SIMPLEX,1,255)
		cv2.imshow('feed',img)
	
	if cv2.waitKey(1) == 27:
		#cv2.imwrite("./pari2.jpg", img[x, y, w, h])	
		break

cam.release()
cv2.destroyAllWindows()