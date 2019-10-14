import face_recognition as recog 
from PIL import Image, ImageDraw

image = recog.load_image_file("./images/inputs/team-4.jpg")
faceLocations = recog.face_locations(image)

print("Number of Faces: "+str(len(faceLocations)))

pilImage = Image.fromarray(image)
draw = ImageDraw.Draw(pilImage)

for top, right, bottom, left in faceLocations:
	draw.rectangle(((left, top), (right, bottom)), outline=(0,0,255))

del draw

pilImage.show()
pilImage.save("./images/outputs/op6.jpg")