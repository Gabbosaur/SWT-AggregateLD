import pickle
import calculateFeatureAndTrain_module
import numpy as np


# ---------- carico miglior modello
model = pickle.load(open("xgboost.sav", 'rb'))
img_path = "C:\\Users\\Orphe\\Desktop\\Bibbia 2\\Semantic Web Technologies\\Progetto_SWT_TFOD\\TFODCourse\\images\\cas2.jpg"
image_feature = calculateFeatureAndTrain_module.singleImageFeatureExtraction(img_path)
image_feature = np.array([image_feature])

# Predict the response for test dataset
y_pred = model.predict(image_feature)
y_pred_proba = model.predict_proba(image_feature)

print("Probabilità: ", y_pred_proba)
print("Varianza: ", np.var(y_pred_proba))
calculateFeatureAndTrain_module.print_model_score(model)

if y_pred == 0:
	print("Blackboard")
elif y_pred == 1:
	print("Slide")
elif y_pred == 2:
	print("Slide and talk")
elif y_pred == 3:
	print("Talk")
else:
	print("ALTRO")
