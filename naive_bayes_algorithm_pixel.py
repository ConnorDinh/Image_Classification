import data_processing
import feature_extraction
import math
DIGIT_IMAGE_SIZE = 784
FACE_IMAGE_SIZE = 4200

class NaiveBayesClassifier:
	def __init__(self, features, labels, number_pixel, number_classes):
		self.features = features
		self.labels = labels
		self.number_pixel = number_pixel
		self.smoothing = 1.0
		self.number_classes = number_classes
		self.number_images = len(labels)

		self.classes_count = [0.0] * self.number_classes
		self.prob_labels = [0.0] * self.number_classes
		self.compute_probability_label()

		self.pixel_count_1 = []
		for i in range(self.number_classes):
			row = []
			for j in range(self.number_pixel):
				row.append(0)
			self.pixel_count_1.append(row)
		self.count_pixel_is_1_for_each_class()

		self.pixel_prob_1 = []
		self.pixel_prob_0 = []
		self.count_number_of_time_pixel_appear_given_label()

		self.pixel_prob_1_not_label = []
		self.pixel_prob_0_not_label = []
		self.count_number_of_time_pixel_appear_given_not_label()


	def compute_probability_label(self):
		for label in self.labels:
			self.classes_count[label] += 1
		for i in range(self.number_classes):
			p = self.classes_count[i]/self.number_images
			if p <= 0.0:
				p = 1e-12
			self.prob_labels[i] = math.log(p)

	def count_pixel_is_1_for_each_class(self):
		for i in range(self.number_images):
			pixels = self.features[i]
			label = self.labels[i]
			for j in range(self.number_pixel):
				if pixels[j] == 1:
					self.pixel_count_1[label][j] += 1

	def count_number_of_time_pixel_appear_given_label(self):
		for i in range(self.number_classes):
			prob_row_1 = []
			prob_row_0 = []
			number_of_class_i = self.classes_count[i]
			for j in range(self.number_pixel):
				count1 = self.pixel_count_1[i][j]
				p1 = (count1 + self.smoothing)/(number_of_class_i + 2*self.smoothing)
				p0 = 1.0-p1
				if p1 <= 0.0:
					p1 = 1e-12
				if p0 <= 0.0:
					p0 = 1e-12
				prob_row_1.append(math.log(p1))
				prob_row_0.append(math.log(p0))
			self.pixel_prob_1.append(prob_row_1)
			self.pixel_prob_0.append(prob_row_0)

	def helper_function(self, class_index, pixel_index):
		total = 0.0
		for i in range(self.number_classes):
			if i != class_index:
				total += self.pixel_count_1[i][pixel_index]
		return total


	def count_number_of_time_pixel_appear_given_not_label(self):
		for i in range(self.number_classes):
			prob_row_1 = []
			prob_row_0 = []
			for j in range(self.number_pixel):
				p1 = (self.helper_function(i, j) + self.smoothing)/(self.number_images - self.classes_count[i] + 2*self.smoothing)
				p0 = 1.0-p1
				if p1 <= 0.0:
					p1 = 1e-12
				if p0 <= 0.0:
					p0 = 1e-12
				prob_row_1.append(math.log(p1))
				prob_row_0.append(math.log(p0))
			self.pixel_prob_1_not_label.append(prob_row_1)
			self.pixel_prob_0_not_label.append(prob_row_0)


	def fit(self, images):
		images_features = feature_extraction.pixels_extraction(images)
		predictions = []
		
		for image_features in images_features:
			products = []
			products_label = []
			products_not_label = []
			
			for i in range(self.number_classes):
				product_label = self.prob_labels[i]
				for j in range(self.number_pixel):
					if image_features[j] == 1:
						product_label = product_label + self.pixel_prob_1[i][j]
					else:
						product_label = product_label + self.pixel_prob_0[i][j]
				products_label.append(product_label)

			for i in range(self.number_classes):
				product_not_label = math.log(1 - math.exp(self.prob_labels[i]))
				for j in range(self.number_pixel):
					if image_features[j] == 1:
						product_not_label = product_not_label + self.pixel_prob_1_not_label[i][j]
					else:
						product_not_label = product_not_label + self.pixel_prob_0_not_label[i][j]
				products_not_label.append(product_not_label)

			for i in range(len(products_label)):
				products.append(products_label[i] - products_not_label[i])

			best_result = 0
			best_product = products[0]
			for i in range(1, self.number_classes):
				if products[i] > best_product:
					best_product = products[i]
					best_result = i
			predictions.append(best_result)
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
		digit_naive = NaiveBayesClassifier(digit_train_features[:end_index_digit], digit_train_labels[:end_index_digit], DIGIT_IMAGE_SIZE, 10)
		face_naive = NaiveBayesClassifier(face_train_features[:end_index_face], face_train_labels[:end_index_face], FACE_IMAGE_SIZE, 2)


		digit_predictions = digit_naive.fit(digit_val_images)
		face_predictions = face_naive.fit(face_val_images)

		digit_acc[i] = compute_accuracy(digit_predictions, digit_val_labels)

		face_acc[i] = compute_accuracy(face_predictions, face_val_labels)

	print("The average test accuracy of digit images: " + str(compute_mean(digit_acc)*100) +" %")
	print("The standard deviation is: +-" + str(round(compute_standard_deviation(digit_acc)*100, 2)))
	print("The average test accuracy of face images: " + str(compute_mean(face_acc)*100) + " %")
	print("The standard deviation is: +-" + str(round(compute_standard_deviation(face_acc)*100, 2)))

	
main()
		
