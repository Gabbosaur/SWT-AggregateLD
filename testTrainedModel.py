import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_PATH = "C:\\Users\\Orphe\\Desktop\\Bibbia 2\\Semantic Web Technologies\\Progetto_SWT_TFOD\\TFODCourse\\images\\test\\talk\\talk54.jpg"
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5) as pose:
	image = cv2.imread(IMAGE_PATH)
	image_height, image_width, _ = image.shape
	# Convert the BGR image to RGB before processing.
	results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


	for i in range(0,33):
		results.pose_landmarks.landmark[i].x = results.pose_landmarks.landmark[i].x * image_width
		results.pose_landmarks.landmark[i].y = results.pose_landmarks.landmark[i].y * image_height
	print("")
	print(results.pose_landmarks)

	if not results.pose_landmarks:
		print("Non trovo pose.")


	annotated_image = image.copy()

	# Draw pose landmarks on the image.
	mp_drawing.draw_landmarks(
		annotated_image,
		results.pose_landmarks,
		mp_pose.POSE_CONNECTIONS,
		landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
	cv2.imwrite('111.png', annotated_image)
	print("Immagine salvata")
	# # Plot pose world landmarks.
	# mp_drawing.plot_landmarks(
	# 	results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)





# import pickle
# import calculateFeatureAndTrain_module
# import numpy as np
# import cv2 as cv

# # ---------- carico miglior modello
# model = pickle.load(open("xgboost.sav", 'rb'))
# img_path = "C:\\Users\\Orphe\\Desktop\\Bibbia 2\\Semantic Web Technologies\\Progetto_SWT_TFOD\\TFODCourse\\images\\test\\talk\\talk52.jpg"

# img = cv.imread(img_path,0)
# image_feature = calculateFeatureAndTrain_module.singleImageFeatureExtraction(img=img)
# image_feature = np.array([image_feature])

# # Predict the response for test dataset
# y_pred = model.predict(image_feature)
# y_pred_proba = model.predict_proba(image_feature)


# print("Probabilità: ", y_pred_proba)
# print("Varianza: ", np.var(y_pred_proba))
# # calculateFeatureAndTrain_module.print_model_score(model)

# print(type(int(y_pred)))

# if y_pred == 0:
# 	print("Blackboard")
# elif y_pred == 1:
# 	print("Slide")
# elif y_pred == 2:
# 	print("Slide and talk")
# elif y_pred == 3:
# 	print("Talk")
# else:
# 	print("ALTRO")
