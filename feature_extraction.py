#import data_processing

DIGIT_IMAGE_SIZE = 784
FACE_IMAGE_SIZE = 4200

def pixels_extraction(images):
	features = []
	for image in images:
		arr_pixels = []
		for i in range(len(image)):
			if image[i] == ' ':
				arr_pixels.append(0)
			else:
				arr_pixels.append(1)
		features.append(arr_pixels)
	return features

def counting_extraction(images):
	features = []
	for image in images:
		whitespace_counting = 0
		symbol_counting = 0
		for i in range(len(image)):
			if image[i] == ' ':
				whitespace_counting += 1
			else:
				symbol_counting += 1
		features.append([whitespace_counting, symbol_counting])
	return features


#def main():
	#digit = data_processing.load_digit_images()
	#pixels_extraction(digit["training"][0][0])
	#print(digit["training"][0][0])
	#tup = counting_extraction(digit["training"][0][0])
	#print(tup[0])
	#print(tup[1])
#main()