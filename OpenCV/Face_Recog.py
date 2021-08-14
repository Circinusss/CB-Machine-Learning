import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data = []
dataset_path = './Faces/'
file_name = input("Enter Your Name: ")

while True:
	ret,frame = cap.read()
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret == False:
		continue

	#Collecting array of face detected
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	faces = sorted(faces,key=lambda f:f[2]*f[3])
	

	#Face Reactangle
	for (x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#Extract Faces
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip += 1

		#Store every 10th face
		if(skip%10==0):
			face_data.append(face_section)
			print(len(face_data))

	cv2.imshow("Video Frame",frame)
	cv2.imshow('Face Section', face_section)

	#To terminate the window
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

#Convert face list into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save data into file
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Saved")

cap.release()
cv2.destroyAllWindows()