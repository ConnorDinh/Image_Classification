DIGIT_IMAGE_SIZE = 784
FACE_IMAGE_SIZE = 4200

#Load labels
def load_labels(path):
	labels = []
	with open(path, "r") as file:
		for line in file:
			labels.append(int(line.strip()))
	return labels

#Load data
def load_data(path):
	data = []
	with open(path, "r") as file:
		for line in file:
			for char in line.rstrip('\n'):
				data.append(char)
	return data

#Load images
def load_images(path, image_size, num_images):
	images = []
	data = load_data(path)
	index = 0
	for i in range(num_images):
		image = []
		for j in range(image_size):
			image.append(data[index])
			index+=1
		images.append(image)
	return images

#Load digit images
def load_digit_images():
	training_labels = load_labels("data/digitdata/traininglabels")
	training_images = load_images("data/digitdata/trainingimages", DIGIT_IMAGE_SIZE, len(training_labels))
	validation_labels = load_labels("data/digitdata/validationlabels")
	validation_images = load_images("data/digitdata/validationimages", DIGIT_IMAGE_SIZE, len(validation_labels))
	test_labels = load_labels("data/digitdata/testlabels")
	test_images = load_images("data/digitdata/testimages", DIGIT_IMAGE_SIZE, len(test_labels))

	return {
	"training": (training_images, training_labels),
	"validation": (validation_images, validation_labels),
	"test": (test_images, test_labels)
	}

#Load face images
def load_face_images():
	training_labels = load_labels("data/facedata/facedatatrainlabels")
	training_images = load_images("data/facedata/facedatatrain", FACE_IMAGE_SIZE, len(training_labels))
	validation_labels = load_labels("data/facedata/facedatavalidationlabels")
	validation_images = load_images("data/facedata/facedatavalidation", FACE_IMAGE_SIZE, len(validation_labels))
	test_labels = load_labels("data/facedata/facedatatestlabels")
	test_images = load_images("data/facedata/facedatatest", FACE_IMAGE_SIZE, len(test_labels))

	return {
	"training": (training_images, training_labels),
	"validation": (validation_images, validation_labels),
	"test": (test_images, test_labels)
	}

#def main():
#	digit = load_digit_images()
#	print(digit["training"][0][0])
#	load_face_images()

#main()
