from flask import Blueprint, request, jsonify
from .model import net, center_image, softmax
import numpy as np
from PIL import Image
from io import BytesIO
import base64


recognizer_bp = Blueprint("recognizer", __name__)

@recognizer_bp.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    pixels = np.array(data['pixels'], dtype=np.float32)
    pixels = center_image(pixels)
    output = net.feedforward(pixels)
    if output.shape != (10,):  # Ensure output is a 1D array with 10 elements
        return jsonify({'error': 'Invalid output shape from network'}), 500
    
    probabilities = softmax(output)
    prediction = np.argmax(probabilities)
    confidences = [{'digit': i, 'prob': round(float(probabilities[i] * 100),2)} for i in range(10)]

    img = Image.fromarray((pixels.reshape(28, 28) * 255).astype(np.uint8), mode='L')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')   

    return jsonify({
        'prediction': int(prediction),
        'confidences': confidences,
        'image': img_base64
    })