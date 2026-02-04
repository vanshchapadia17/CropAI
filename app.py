from flask import Flask, render_template, request, os
from werkzeug.utils import secure_filename
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

# Configure where to save uploaded photos
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file selected')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', message='No file selected')

        if file:
            # 2. Save the file to the uploads folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 3. Use your Pipeline to get the prediction
            pipeline = PredictPipeline()
            # Note: We need to map the number (0, 1, 2) back to a name
            crop_map = {0: "Jute", 1: "Maize", 2: "Rice", 3: "Sugarcane", 4: "Wheat"}
            
            prediction_idx = pipeline.predict(file_path)
            result = crop_map.get(prediction_idx, "Unknown")

            return render_template('index.html', 
                                 prediction=result, 
                                 image_path=file_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)