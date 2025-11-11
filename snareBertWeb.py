import os
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename

from snareBertQrDetect import detect_and_decode_qr_with_qrdet, BertForURLClassification, preprocess_url, predict_url
import torch
from transformers import BertTokenizer

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Model Loading ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "bert_model/BERT.pt"
CONFIG_PATH = "bert_config/config.json"
VOCAB_DIR = "bert_tokenizer/"
FINETUNED_CHECKPOINT = "bert_finetuned/fine_tuned_phishing_bert_ep4.pth"

print("Loading model...")
tokenizer = BertTokenizer.from_pretrained(VOCAB_DIR, do_lower_case=False)
model = BertForURLClassification(MODEL_PATH, CONFIG_PATH)
model.load_state_dict(torch.load(FINETUNED_CHECKPOINT, map_location=DEVICE), strict=False)
model.to(DEVICE)
model.eval()
print("Model loaded.")
# --- End Model Loading ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('upload.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', error='No selected file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        decoded_urls = detect_and_decode_qr_with_qrdet(filepath, model, tokenizer, DEVICE)
        
        results = []
        for url in decoded_urls:
            prediction = predict_url(url, model, tokenizer, DEVICE)
            results.append({'url': url, 'prediction': prediction})

        detections_filename = None
        if decoded_urls:
            base, ext = os.path.splitext(filename)
            detections_filename = f"{base}_detections{ext}"

        return render_template('results.html', results=results, original_image=filename, detections_image=detections_filename)
    return render_template('upload.html', error='File type not allowed')

@app.route('/api/predict_url', methods=['POST'])
def predict_scanned_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL not provided'}), 400
    
    url = data['url']
    prediction = predict_url(url, model, tokenizer, DEVICE)
    return jsonify(prediction)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    context = ('static/cert.pem', 'static/key.pem')
    app.run(host='0.0.0.0', port=5000, ssl_context=context, debug=True)
