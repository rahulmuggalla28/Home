# pip install flask numpy pandas scikit-learn

from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open('House.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def predict():
    bed = int(request.form['bedrooms'])
    bath = int(request.form['bathrooms'])
    loc = int(request.form['location'])
    size = int(request.form['size'])
    status = int(request.form['status'])
    face = int(request.form['facing'])
    Type = int(request.form['type'])
    
    input_data = np.array([[bed, bath, loc, size,
                            status, face, Type]])
    
    prediction = model.predict(input_data)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
