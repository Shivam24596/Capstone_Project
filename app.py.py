from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load your model (make sure it's trained on 5 features)
model = pickle.load(open('delivery_model.pkl', 'rb'))

# Encoding maps (must match model training)
shipping_mode_map = {
    'First Class': 0,
    'Second Class': 1,
    'Standard Class': 2,
    'Same Day': 3
}

order_region_map = {
    'Southeast Asia': 0,
    'South Asia': 1,
    'Oceania': 2,
    'Eastern Asia': 3,
    'West Asia': 4,
    'US Center': 5,
    'West of USA': 6,
    'East of USA': 7,
    'Canada': 8,
    'Western Europe': 9
}

order_state_map = {
    'California': 0,
    'New York': 1,
    'Texas': 2,
    'Ontario': 3,
    'Florida': 4,
    'Maharashtra': 5,
    'Tokyo': 6,
    'Delhi': 7,
    'Seoul': 8,
    'Quebec': 9
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        x = [
            shipping_mode_map[request.form['shipping_mode']],
            order_region_map[request.form['order_region']],
            order_state_map[request.form['order_state']],
            int(request.form['quantity']),
            float(request.form['price'])
        ]

        prediction = model.predict([x])[0]

        #  Save the improved horizontal bar graph
        plt.figure(figsize=(6, 2))
        plt.barh(['Predicted Delivery Time'], [prediction], color='skyblue')
        plt.xlabel("Days")
        plt.xlim(0, 15)  # Adjust max based on your model's range
        plt.title("Predicted Delivery Time (in Days)")
        for i, v in enumerate([prediction]):
            plt.text(v + 0.2, i, f"{v:.2f} days", va='center')
        plt.tight_layout()

        if not os.path.exists('static'):
            os.makedirs('static')
        plt.savefig('static/plot.png')
        plt.close()

        return render_template('index.html', result=f"📦 Estimated Delivery Time: {prediction:.2f} days")

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
