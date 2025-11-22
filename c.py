from flask import Flask, render_template_string, request, jsonify
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os

app = Flask(__name__)

# Configuration
MIN_CONFIDENCE = 0.40
MAX_CONFIDENCE = 1.00
DEVICE = torch.device('cpu')

# ImageNet classes
imagenet_classes = {
    "0": "tench", "1": "goldfish", "2": "great white shark", "3": "tiger shark",
    "4": "hammerhead", "5": "electric ray", "6": "stingray", "7": "rooster",
    "8": "hen", "9": "ostrich", "10": "brambling", "11": "goldfinch",
    "99": "goose", "151": "Chihuahua", "152": "Japanese spaniel",
    "153": "Maltese dog", "154": "Pekinese", "155": "Shih-Tzu",
    "156": "Blenheim spaniel", "157": "papillon", "158": "toy terrier",
    "207": "golden retriever", "208": "Labrador retriever",
    "282": "tiger cat", "283": "Persian cat", "284": "Siamese cat",
    "291": "lion", "292": "tiger", "293": "cheetah", "294": "brown bear",
    "404": "airliner", "407": "ambulance", "429": "baseball", "430": "basketball",
    "504": "coffee mug", "620": "laptop", "850": "television", "858": "toaster",
    "927": "ice cream", "948": "strawberry", "949": "orange", "950": "lemon", "962": "pizza"
}

print("Loading ResNet50 model...")
try:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = models.resnet50(pretrained=True)
    model = model.to(DEVICE)
    model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_bytes):
    """Predict objects - returns 40% to 100% confidence"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
        
        top_probs, top_ids = torch.topk(probs, 50)
        
        predictions = []
        for i in range(top_probs.size(0)):
            conf = top_probs[i].item()
            if MIN_CONFIDENCE <= conf <= MAX_CONFIDENCE:
                class_id = str(top_ids[i].item())
                class_name = imagenet_classes.get(class_id, f"Class {class_id}")
                conf_percent = round(conf * 100, 1)
                predictions.append({
                    'class': class_name,
                    'confidence': conf_percent
                })
            elif conf < MIN_CONFIDENCE:
                break
        
        return predictions
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        image_bytes = file.read()
        predictions = predict_image(image_bytes)
        
        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <title>AI Object Recognition - 40% to 100%</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 18px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .subtitle {
            color: #666;
            font-size: 1.1em;
            margin-top: 10px;
        }
        
        .info-box {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
            color: #1565c0;
            font-weight: 600;
            border-left: 5px solid #1565c0;
            font-size: 1.05em;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 50px 30px;
            text-align: center;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 25px;
        }
        
        .upload-area:hover {
            background: linear-gradient(135deg, #e0e7ff 0%, #b3c6ff 100%);
            border-color: #667eea;
            transform: scale(1.01);
        }
        
        .upload-area.dragover {
            background: #d8e3ff;
            border-color: #667eea;
            transform: scale(1.02);
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 15px;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 13px 32px;
            font-size: 1em;
            border-radius: 50px;
            cursor: pointer;
            margin: 10px;
            transition: all 0.3s ease;
            font-weight: bold;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .btn:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        .preview-section {
            margin-top: 30px;
            display: none;
        }
        
        .image-preview {
            text-align: center;
            margin-bottom: 25px;
        }
        
        .preview-img {
            max-width: 100%;
            max-height: 350px;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        }
        
        .predictions {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 25px;
        }
        
        .predictions h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            text-align: center;
        }
        
        .prediction-item {
            background: white;
            padding: 18px;
            margin: 12px 0;
            border-radius: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .prediction-item:hover {
            transform: translateX(8px);
            box-shadow: 0 6px 18px rgba(102, 126, 234, 0.2);
        }
        
        .class-name {
            font-size: 1.2em;
            color: #333;
            font-weight: 600;
            text-transform: capitalize;
            flex-grow: 1;
        }
        
        .confidence {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 10px 22px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.1em;
            min-width: 100px;
            text-align: center;
            box-shadow: 0 3px 10px rgba(17, 153, 142, 0.3);
        }
        
        .loading {
            text-align: center;
            padding: 50px;
            display: none;
        }
        
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            width: 70px;
            height: 70px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading p {
            font-size: 1.2em;
            color: #667eea;
            font-weight: bold;
        }
        
        .no-predictions {
            text-align: center;
            padding: 30px;
            color: #d32f2f;
            font-size: 1.1em;
            font-weight: 500;
        }
        
        .model-info {
            background: linear-gradient(135deg, #f3e5f5 0%, #ede7f6 100%);
            padding: 18px;
            border-radius: 12px;
            margin-top: 30px;
            text-align: center;
            color: #6a1b9a;
            font-weight: 600;
            border-left: 5px solid #6a1b9a;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Object Recognition</h1>
            <div class="badge">ResNet50 Deep Learning</div>
            <p class="subtitle">Upload image and detect objects</p>
        </div>
        
        <div class="info-box">
            📊 Confidence Range: 40% - 100% | All detected objects in this range
        </div>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">🖼️</div>
            <h2>Click or Drag & Drop Image</h2>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        
        <div style="text-align: center;">
            <button class="btn" onclick="document.getElementById('fileInput').click()">
                📁 Choose Image
            </button>
            <button class="btn" id="predictBtn" onclick="predictImage()" disabled>
                🔮 Predict
            </button>
            <button class="btn btn-secondary" onclick="resetApp()">
                🔄 Reset
            </button>
        </div>
        
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>Analyzing image...</p>
        </div>
        
        <div id="previewSection" class="preview-section">
            <div class="image-preview">
                <img id="previewImg" class="preview-img" src="" alt="Preview">
            </div>
            
            <div class="predictions">
                <h2>🎯 Detected Objects (40% - 100%)</h2>
                <div id="predictionsContainer"></div>
            </div>
        </div>
        
        <div class="model-info">
            💡 ResNet50 on 1000+ ImageNet categories. Displays all predictions from 40% to 100% confidence.
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const predictBtn = document.getElementById('predictBtn');
        const previewSection = document.getElementById('previewSection');
        const previewImg = document.getElementById('previewImg');
        const loading = document.getElementById('loading');
        const predictionsContainer = document.getElementById('predictionsContainer');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please upload an image file!');
                return;
            }
            
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                uploadArea.innerHTML = `
                    <div class="upload-icon">✅</div>
                    <h2>Image Ready!</h2>
                    <p style="color: #667eea; margin-top: 8px; font-weight: bold;">${file.name}</p>
                `;
                predictBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
        
        async function predictImage() {
            if (!selectedFile) return;
            
            loading.style.display = 'block';
            previewSection.style.display = 'none';
            predictBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayPredictions(data.predictions);
                    previewSection.style.display = 'block';
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Prediction failed: ' + error.message);
            } finally {
                loading.style.display = 'none';
                predictBtn.disabled = false;
            }
        }
        
        function displayPredictions(predictions) {
            if (predictions.length === 0) {
                predictionsContainer.innerHTML = '<div class="no-predictions">❌ No objects detected above 40% confidence</div>';
                return;
            }
            
            let html = '';
            predictions.forEach((pred) => {
                html += `
                    <div class="prediction-item">
                        <span class="class-name">${pred.class}</span>
                        <span class="confidence">${pred.confidence}%</span>
                    </div>
                `;
            });
            predictionsContainer.innerHTML = html;
        }
        
        function resetApp() {
            selectedFile = null;
            fileInput.value = '';
            previewImg.src = '';
            previewSection.style.display = 'none';
            loading.style.display = 'none';
            predictBtn.disabled = true;
            uploadArea.innerHTML = `
                <div class="upload-icon">🖼️</div>
                <h2>Click or Drag & Drop Image</h2>
            `;
        }
    </script>
</body>
</html>
'''