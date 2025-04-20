import os
import numpy as np
import tensorflow as tf
from tensorflow import keras  # Add explicit keras import
from keras.utils import get_file  # Add get_file import
from keras.preprocessing import image  # Add image preprocessing import
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
import datetime
import json
import io
import base64
import pathlib
import cv2  # Add OpenCV import

# Initialize Flask application
app = Flask(__name__, static_folder='static')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}
app.config['HISTORY_FILE'] = 'data/history.json'
app.config['MODEL_PATH'] = 'model/model.h5'  # Updated to use the correct model file

# Ensure required directories exist
for directory in [app.config['UPLOAD_FOLDER'], 'data', 'static/images']:
    os.makedirs(directory, exist_ok=True)

# Create history file if it doesn't exist
if not os.path.exists(app.config['HISTORY_FILE']):
    with open(app.config['HISTORY_FILE'], 'w') as f:
        json.dump([], f)

# Fruit categories for classification based on the dataset
FRUIT_CATEGORIES = ['apple', 'banana', 'orange']
CLASSES = ['Fresh Fruit', 'Rotten Fruit']  # Binary classification classes

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the trained TensorFlow model"""
    try:
        # Use keras directly instead of tf.keras
        model_predict = keras.models.load_model(app.config['MODEL_PATH'])
        # Compile the model with appropriate parameters
        model_predict.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        print("Model loaded and compiled successfully!")
        return model_predict
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model at startup
model = load_model()

def preprocess_image(image_path, target_size=(150, 150)):
    """Preprocess image for model prediction using keras preprocessing"""
    try:
        # Use keras image preprocessing as in the example code
        img = image.load_img(image_path, color_mode="rgb", 
                            target_size=target_size, 
                            interpolation="nearest")
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255.0
        
        # Stack images as in the example code
        images = np.vstack([img])
        return images
    except Exception as e:
        print(f"Error in keras preprocessing: {e}")
        # Fallback to original PIL method if keras preprocessing fails
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.asarray(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

def detect_fruit_type(image_path):
    """Detect fruit type based on filename or simple visual cues"""
    # This is a simplified implementation - in a real app, you'd use another model
    # Just randomly assign a fruit type for now
    return np.random.choice(FRUIT_CATEGORIES)

def get_recommendations(is_fresh, fruit_type):
    """Generate recommendations based on freshness and fruit type"""
    if is_fresh:
        recommendations = {
            'apple': "This apple appears fresh. Store in a cool, dry place or refrigerate to maintain freshness for up to 4-6 weeks.",
            'banana': "This banana is fresh. Store at room temperature away from direct sunlight. Will ripen in 2-5 days.",
            'orange': "This orange is fresh. Store in a cool, dry place or refrigerate for up to 3 weeks."
        }.get(fruit_type, "This fruit appears fresh. Store appropriately based on the fruit type.")
    else:
        recommendations = {
            'apple': "This apple shows signs of spoilage. Discard if there's significant mold or unusual odors. Small bruised areas can be cut away.",
            'banana': "This banana appears overripe or spoiled. If only brown/black on the outside, it may still be good for banana bread. Discard if there's mold.",
            'orange': "This orange shows signs of spoilage. Check for mold, unusual softness, or off odors. Discard if significantly compromised."
        }.get(fruit_type, "This fruit appears spoiled. It's recommended to discard it to avoid potential health risks.")
    
    return recommendations

def get_shelf_life_info(fruit_type):
    """Get shelf life information for various fruit types"""
    shelf_life = {
        'apple': {
            "refrigerated": "4-6 weeks", 
            "room_temp": "1-2 weeks",
            "signs_of_spoilage": "Soft spots, wrinkled skin, discoloration, mold, off odor"
        },
        'banana': {
            "refrigerated": "5-7 days (skin will darken)", 
            "room_temp": "2-5 days when ripe",
            "signs_of_spoilage": "Mold, fruit flies, liquid leakage, fermented smell"
        },
        'orange': {
            "refrigerated": "2-3 weeks", 
            "room_temp": "1 week",
            "signs_of_spoilage": "Mold, soft spots, discoloration, sour smell"
        }
    }.get(fruit_type, {"general": "Varies by fruit type"})
    
    return shelf_life

def add_to_history(image_filename, result, confidence, fruit_type):
    """Add analysis result to history"""
    try:
        with open(app.config['HISTORY_FILE'], 'r') as f:
            history = json.load(f)
        
        # Create new history entry
        entry = {
            'id': str(uuid.uuid4()),
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image': image_filename,
            'result': result,
            'confidence': round(float(confidence) * 100, 2),  # Convert to percentage
            'fruit_type': fruit_type
        }
        
        # Add to history and limit to most recent 50 entries
        history.append(entry)
        history = history[-50:]
        
        with open(app.config['HISTORY_FILE'], 'w') as f:
            json.dump(history, f)
            
        return True
    except Exception as e:
        print(f"Error adding to history: {e}")
        return False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze fruit image for freshness/spoilage"""
    # Check if model is loaded
    global model
    if model is None:
        model = load_model()
        if model is None:
            return jsonify({
                'error': 'Model not available. Please try again later.'
            }), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        print("Error: No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        print("Error: Empty filename submitted")
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        print(f"Error: File type not allowed - {file.filename}")
        return jsonify({'error': f'File type not allowed. Allowed types: {app.config["ALLOWED_EXTENSIONS"]}'}), 400

    try:
        # Save file with secure filename
        filename = secure_filename(file.filename)
        # Add timestamp to filename to avoid duplicates
        filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Detect fruit type (in a real app, this would use another model)
        fruit_type = detect_fruit_type(file_path)
        
        # Preprocess image
        img_array = preprocess_image(file_path)
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]  # Our model gives a single value between 0 and 1
        
        # For binary classification: close to 0 is Fresh, close to 1 is Rotten
        # Adjust if your model has different interpretation
        is_fresh = prediction < 0.5
        result = "Fresh Fruit" if is_fresh else "Rotten Fruit"
        confidence = 1 - prediction if is_fresh else prediction
        
        # Get recommendations
        recommendation = get_recommendations(is_fresh, fruit_type)
        
        # Get shelf life info
        shelf_life = get_shelf_life_info(fruit_type)
        
        # Add to history
        add_to_history(filename, result, confidence, fruit_type)
        
        # Return result
        return jsonify({
            'result': result,
            'confidence': round(float(confidence) * 100, 2),  # Convert to percentage
            'fruit_type': fruit_type,
            'recommendation': recommendation,
            'shelf_life': shelf_life,
            'image_url': f'/uploads/{filename}'
        })
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-camera', methods=['POST'])
def analyze_camera():
    """Analyze fruit image from camera capture"""
    # Check if base64 image was sent
    if 'image_data' not in request.json:
        return jsonify({'error': 'No image data'}), 400
    
    try:
        # Get image data
        image_data = request.json['image_data'].split(',')[1]  # Remove data URL prefix
        image_bytes = io.BytesIO(base64.b64decode(image_data))
        
        # Save image
        filename = f"camera_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(file_path, 'wb') as f:
            f.write(image_bytes.getvalue())
        
        # Detect fruit type
        fruit_type = detect_fruit_type(file_path)
        
        # Preprocess and predict
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)[0][0]
        
        # For binary classification: close to 0 is Fresh, close to 1 is Rotten
        is_fresh = prediction < 0.5
        result = "Fresh Fruit" if is_fresh else "Rotten Fruit"
        confidence = 1 - prediction if is_fresh else prediction
        
        recommendation = get_recommendations(is_fresh, fruit_type)
        shelf_life = get_shelf_life_info(fruit_type)
        
        add_to_history(filename, result, confidence, fruit_type)
        
        return jsonify({
            'result': result,
            'confidence': round(float(confidence) * 100, 2),
            'fruit_type': fruit_type,
            'recommendation': recommendation,
            'shelf_life': shelf_life,
            'image_url': f'/uploads/{filename}'
        })
        
    except Exception as e:
        print(f"Error during camera analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    try:
        with open(app.config['HISTORY_FILE'], 'r') as f:
            history = json.load(f)
        return jsonify(history)
    except Exception as e:
        print(f"Error getting history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear analysis history"""
    try:
        with open(app.config['HISTORY_FILE'], 'w') as f:
            json.dump([], f)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error clearing history: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large (max 16MB)'}), 413

@app.errorhandler(500)
def server_error(error):
    """Handle server errors"""
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    # Return a simple text response instead of template while 404.html doesn't exist
    return "Page not found", 404

if __name__ == '__main__':
    # Use debug=True during development, set to False in production
    app.run(debug=True, host='0.0.0.0', port=5000)