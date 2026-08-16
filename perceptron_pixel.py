import data_processing
import feature_extraction
import math
DIGIT_IMAGE_SIZE = 784
FACE_IMAGE_SIZE = 4200

class Perceptron:
	def __init__(self, features, labels, number_pixel, number_classes):
		self.features = features
		self.labels = labels
		self.number_pixel = number_pixel
		self.number_classes = number_classes
		self.number_images = len(labels)
		self.bias = 1
		self.learning_time = 4

		self.weight = []
		for i in range(number_classes):
			row_weight = []
			for j in range(number_pixel):
				row_weight.append(0.0)
			self.weight.append(row_weight)

		self.weight0 = []
		for i in range(number_classes):
			self.weight0.append(0.0)

		self.train()

	def compute_function(self, image_features):
		functions = []
		for i in range(self.number_classes):
			function = self.weight0[i]
			for j in range(self.number_pixel):
				function += self.weight[i][j]*image_features[j]
			functions.append(function)
		return functions

	def train(self):
		for k in range(self.learning_time):
			for i in range(self.number_images):
				functions = self.compute_function(self.features[i])

				largest_function = functions[0]
				largest_index = 0
				for j in range(1, self.number_classes):
					if functions[j] >largest_function:
						largest_function = functions[j]
						largest_index = j

				if self.labels[i] != largest_index:
					for j in range(self.number_pixel):
						self.weight[self.labels[i]][j] += self.bias*self.features[i][j]
						self.weight[largest_index][j] -= self.bias*self.features[i][j]
					self.weight0[self.labels[i]] += self.bias
					self.weight0[largest_index] -= self.bias

	def fit(self, images):
		images_features = feature_extraction.pixels_extraction(images)
		predictions = []
		for image_features in images_features:
			functions = self.compute_function(image_features)

			largest_function = functions[0]
			largest_index = 0
			for i in range(1, self.number_classes):
				if functions[i] >largest_function:
					largest_function = functions[i]
					largest_index = i

			predictions.append(largest_index)
		return predictions

def compute_accuracy(predictions, labels):
	count =0.0
	for i in range(len(labels)):
		if predictions[i] == labels[i]:
			count+=1
	return count/len(labels)

def compute_mean(arr):
	total = 0.0
	for i in arr:
		total += i
	return total/len(arr)

def compute_standard_deviation(arr):
	mean = compute_mean(arr)
	variance = 0.0
	for i in arr:
		variance += (i - mean)**2
	variance = variance / len(arr)
	return math.sqrt(variance)


def main():
	digit = data_processing.load_digit_images()
	face = data_processing.load_face_images()

	digit_train_images = digit["training"][0]
	digit_train_labels = digit["training"][1]
	digit_val_images = digit["test"][0]
	digit_val_labels = digit["test"][1]

	face_train_images = face["training"][0]
	face_train_labels = face["training"][1]
	face_val_images = face["test"][0]
	face_val_labels = face["test"][1]
	
	digit_train_features = feature_extraction.pixels_extraction(digit_train_images)
	face_train_features = feature_extraction.pixels_extraction(face_train_images)

	digit_acc = [0.0] *5
	face_acc = [0.0] *5

	for i in range(5):
		end_index_digit = int((0.2+i*0.2)*len(digit_train_features))
		end_index_face = int((0.2+i*0.2)*len(face_train_features))
		digit_perceptron = Perceptron(digit_train_features[:end_index_digit], digit_train_labels[:end_index_digit], DIGIT_IMAGE_SIZE, 10)
		face_perceptron = Perceptron(face_train_features[:end_index_face], face_train_labels[:end_index_face], FACE_IMAGE_SIZE, 2)


		digit_predictions = digit_perceptron.fit(digit_val_images)
		face_predictions = face_perceptron.fit(face_val_images)

		digit_acc[i] = compute_accuracy(digit_predictions, digit_val_labels)

		face_acc[i] = compute_accuracy(face_predictions, face_val_labels)

	print(digit_acc)

	print("The average test accuracy of digit images: " + str(compute_mean(digit_acc)*100) +" %")
	print("The standard deviation is: +-" + str(round(compute_standard_deviation(digit_acc)*100, 2)))
	print("The average test accuracy of face images: " + str(compute_mean(face_acc)*100) + " %")
	print("The standard deviation is: +-" + str(round(compute_standard_deviation(face_acc)*100, 2)))


main()