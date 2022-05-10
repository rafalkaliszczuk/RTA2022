from flask import Flask, request
import numpy as np
import pickle


app = Flask(__name__)

@app.route('/predict')
def predict():

	sep_len = request.args['sepal_length']
	sep_wid = request.args['sepal_width']
	pet_len = request.args['petal_length']
	pet_wid = request.args['petal_width']
	
	test_data = np.array([sep_len, sep_wid, pet_len, pet_wid]).reshape(1,4)
	class_prediced = int(perceptron_model.predict(test_data)[0])
	output = "Predicted Iris Class: " + str(class_prediced)
	
	return (output)

def load_model():
	global perceptron_model
	
	perceptron_file = open('model.pkl', 'rb')
	perceptron_model = pickle.load(perceptron_model)
	perceptron_file.close()

if __name__ == "__main__":
	
	load_model()
	
    app.run()