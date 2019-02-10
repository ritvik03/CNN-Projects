# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])
print("[INFO] network loaded now predicting...")

# classify the input image
(bees, birds, lions, cows) = model.predict(image)[0]
# build the label
#label = "Bees" if bees > birds else "Birds"
proba = max(bees, birds, lions, cows)
if proba==bees:
	    label = "Bees"
elif proba==birds:
	    label="Birds"
elif proba==lions:
	    label="Lions"
elif proba==cows:
	    label="Cows"
#proba = bees if bees > birds else birds
label = "{}: {:.2f}%".format(label, proba * 100)
print(label)

# draw the label on the image
#output = imutils.resize(orig, width=400)
#cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
#	0.7, (0, 255, 0), 2)

# show the output image
#cv2.imshow("Output", output)
#cv2.waitKey(0)
