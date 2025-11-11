import cv2
from pyzbar.pyzbar import decode as pyzbar_decode
from qrdet import QRDetector
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
#
from transformers import AutoConfig, AutoModelForMaskedLM, BertTokenizer

# Define the BertForURLClassification model
class BertForURLClassification(nn.Module):
    def __init__(self, bert_model_path, config_path, num_labels=2):
        super(BertForURLClassification, self).__init__()
        config_kwargs = {
            "cache_dir": None,
            "revision": 'main',
            "use_auth_token": None,
            "hidden_dropout_prob": 0.1,
            "vocab_size": 5000,
        }
        config = AutoConfig.from_pretrained(config_path, **config_kwargs)
        
        self.bert = AutoModelForMaskedLM.from_config(config=config)
        self.bert.cls = nn.Identity()
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        pooled_output = outputs.hidden_states[-1][:,0,:]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

MAX_PREDICT_LEN = 150

def preprocess_url(url, tokenizer, max_len=MAX_PREDICT_LEN):
    tokens = tokenizer.tokenize(url)
    tokens = ["[CLS]"] + tokens[:max_len-2] + ["[SEP]"]
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)
    
    padding_length = max_len - len(input_ids)
    input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    
    return {
        'input_ids': torch.tensor([input_ids]),
        'attention_mask': torch.tensor([attention_mask]),
        'token_type_ids': torch.tensor([token_type_ids])
    }

def predict_url(url, model, tokenizer, device):
    inputs = preprocess_url(url, tokenizer, max_len=MAX_PREDICT_LEN)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    labels = ['benign', 'malicious']
    return {
        'prediction': labels[prediction],
        'probabilities': {
            'benign': probabilities[0][0].item(),
            'malicious': probabilities[0][1].item()
        }
    }

def detect_and_decode_qr_with_qrdet(image_path, model, tokenizer, device):
    """
    Detects QR codes in an image using qrdet and decodes them using pyzbar.

    Args:
        image_path (str): The path to the image file.

    Returns:
        list: A list of decoded URLs found in the QR codes.
              Returns an empty list if no QR codes are found or decoded.
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return []

        # --- Add padding to the image ---
        margin = 50  # Pixels of white space to add around the image
        height, width, _ = img.shape
        new_height = height + 2 * margin
        new_width = width + 2 * margin
        # Create a new image with a white background
        padded_img = np.full((new_height, new_width, 3), 255, dtype=np.uint8)
        # Paste the original image into the center of the new image
        padded_img[margin:margin+height, margin:margin+width] = img

        # Initialize QRDet detector
        detector = QRDetector(model_size='s') 
        print(f"Detecting QR codes in {image_path} using qrdet...")
        detections = detector.detect(image=padded_img, is_bgr=True)

        urls_found = []
        img_display = padded_img.copy()
        found_and_decoded_something = False

        if detections:
            print(f"qrdet found {len(detections)} potential QR code(s).")
            found_and_decoded_something = True
            
            # Draw all qrdet detections
            print("\nVisualizing qrdet detections (bounding boxes):")
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection['bbox_xyxy'])
                confidence = detection['confidence']
                cv2.rectangle(img_display, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                # Create a label with a background
                label = f'{confidence:.2f}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1

                # Get the size of the text box
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Draw a filled rectangle as a background for the text
                cv2.rectangle(img_display, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), cv2.FILLED)

                # Put the text on top of the background rectangle
                cv2.putText(img_display, label, (x1, y1 - 5), 
                            fontFace=font, fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

            decoded_objects = pyzbar_decode(padded_img)

            if decoded_objects:
                print(f"pyzbar decoded {len(decoded_objects)} QR code(s) from the image:")
                for obj in decoded_objects:
                    qr_data = obj.data.decode('utf-8')
                    print(f"  - Type: {obj.type}, Data: {qr_data}")
                    urls_found.append(qr_data)
                    
                    # Predict URL safety
                    prediction_result = predict_url(qr_data, model, tokenizer, device)
                    print(f"URL Safety Prediction: {prediction_result['prediction']}")
                    print(f"Confidence: Benign={prediction_result['probabilities']['benign']:.4f}, "
                          f"Malicious={prediction_result['probabilities']['malicious']:.4f}")
            else:
                print("qrdet found QR regions, but pyzbar could not decode them.")
        else:
            print(f"No QR codes found by qrdet in {image_path}.")
        
        # Save the image with detections if any were made by qrdet
        if found_and_decoded_something:
            base, ext = os.path.splitext(image_path)
            output_image_path = f"{base}_detections{ext}"
            try:
                cv2.imwrite(output_image_path, img_display)
                print(f"Detection visualization saved to: {output_image_path}")
            except Exception as e_save:
                print(f"Error saving detection image: {e_save}")
        
        return urls_found

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect QR codes and predict URL safety.")
    parser.add_argument("image_path", type=str, help="Path to the image file containing QR codes.")
    parser.add_argument("--model_path", type=str, default="bert_model/BERT.pt", help="Path to the pre-trained BERT model weights")
    parser.add_argument("--config_path", type=str, default="bert_config/config.json", help="Path to the BERT model config.json")
    parser.add_argument("--vocab_dir", type=str, default="bert_tokenizer/", help="Path to the directory containing tokenizer's vocab.txt")
    parser.add_argument("--finetuned_checkpoint", type=str, default="bert_finetuned/fine_tuned_phishing_bert_ep4.pth", help="Path to a fine-tuned classification model checkpoint")
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.vocab_dir, do_lower_case=False)
    
    # Load model
    model = BertForURLClassification(args.model_path, args.config_path)
    model.load_state_dict(torch.load(args.finetuned_checkpoint, map_location=device), strict=False)
    model.to(device)
    model.eval()
    
    # Process image and predict URLs
    decoded_urls = detect_and_decode_qr_with_qrdet(args.image_path, model, tokenizer, device)
    
    if decoded_urls:
        print("\nSuccessfully Decoded URLs:")
        for url in decoded_urls:
            print(url)
