from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import numpy as np
import cv2
import sys
import detectar
from extract_overlapped_patches import extract_overlapped_patches
import random
import model

app = Flask(__name__, template_folder='views')
CORS(app)


@app.route('/imagen')
def index():
    return render_template('index.html')

@app.route('/imagen', methods=['POST'])
def procesar_imagen():
    json_data = request.get_json()
    
    base64_image = json_data['imagen']
    base64_image += "=" * ((4 - len(base64_image) % 4) % 4)

    header, data = base64_image.split(',', 1)
    decoded_image = base64.b64decode(data)
    numpy_immage = np.frombuffer(decoded_image, dtype=np.uint8)

    cv_image = cv2.imdecode(numpy_immage, cv2.IMREAD_COLOR)

    # Check if cv_image resolution is bigger than 640 x 640
    if cv_image.shape[0] > 640 or cv_image.shape[1] > 640:
        patches = extract_overlapped_patches(0.2, cv_image)
        # Iterate over dict patches
        for pcenterxy, pimage in patches.items():
            # Detect cells
            img = detectar.detect(pimage, 'yolov7.pt',
                                  patch_center=pcenterxy, glob_img=cv_image)

    

    _, result_base64 = cv2.imencode('.png', img)
    result_base64 = base64.b64encode(result_base64).decode()
    result_base64 = f"data:image/png;base64,{result_base64}"

    return jsonify({'imagen':  result_base64})

if __name__ == '__main__':
    app.run(debug=False)