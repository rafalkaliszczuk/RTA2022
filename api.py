from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return "Hi, Welcome to Flask!!"

@app.route('/predict', methods=['GET'])
def predict():

    sep_len = float(request.args.get('sepal_length'))
    sep_wid = float(request.args.get('sepal_width'))
    pet_len = float(request.args.get('petal_length'))
    pet_wid = float(request.args.get('petal_width'))
    
    
    test_data = [sep_len, sep_wid, pet_len, pet_wid]
    
    perceptron_file = open('model.pkl', 'rb')
    perceptron = pickle.load(perceptron_file)
    perceptron_file.close()
    
    #/predict?sepal_length=4.5&sepal_width=2.3&petal_length=1.3&petal_width=0.3
    prediction = int(perceptron.predict([test_data]))

    return jsonify(features=test_data, predicted_class=prediction)

if __name__ == "__main__":
    app.run(port=5001)
