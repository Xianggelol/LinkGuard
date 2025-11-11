import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM, BertTokenizer
import argparse
import os # Add this import for path manipulation

# Define the classification model structure (adapted from the notebook)
class BertForURLClassification(nn.Module):
    def __init__(self, bert_model_path, config_path, num_labels=2):
        super(BertForURLClassification, self).__init__()
        config_kwargs = {
            "cache_dir": None,
            "revision": 'main',
            "use_auth_token": None,
            "hidden_dropout_prob": 0.1,
            "vocab_size": 5000
        }
        config = AutoConfig.from_pretrained(config_path, **config_kwargs)
        
        self.bert = AutoModelForMaskedLM.from_config(config=config)
        
        # Replace the MLM head with a classification head
        self.bert.cls = nn.Identity() # Remove the MLM head's layers
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

MAX_PREDICT_LEN = 150 # Match fine-tuning MAX_LEN

def preprocess_url(url, tokenizer, max_len=MAX_PREDICT_LEN):
    tokens = tokenizer.tokenize(url)
    tokens = ["[CLS]"] + tokens[:max_len-2] + ["[SEP]"]
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)
    
    # Padding
    padding_length = max_len - len(input_ids)
    input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    
    return {
        'input_ids': torch.tensor([input_ids]),
        'attention_mask': torch.tensor([attention_mask]),
        'token_type_ids': torch.tensor([token_type_ids])
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect URL using BERT")
    parser.add_argument("url", type=str, help="The URL to classify")
    parser.add_argument("--model_path", type=str, default="bert_model/BERT.pt", help="Path to the pre-trained BERT model weights (.pt file)")
    parser.add_argument("--config_path", type=str, default="bert_config/config.json", help="Path to the BERT model config.json")
    parser.add_argument("--vocab_dir", type=str, default="bert_tokenizer/", help="Path to the directory containing tokenizer's vocab.txt")
    parser.add_argument("--finetuned_checkpoint", type=str, default=None, help="Optional path to a fine-tuned classification model checkpoint (e.g., from phishing.ipynb)")

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer from the directory specified by args.vocab_dir
    tokenizer = BertTokenizer.from_pretrained(args.vocab_dir, do_lower_case=False) # Match training

    config = AutoConfig.from_pretrained(args.config_path) # Ensure this config is the one for the fine-tuned model

    class MyFineTunedBert(nn.Module):
        def __init__(self, config_path_for_init, num_labels=2):
            super().__init__()
            l_config = AutoConfig.from_pretrained(config_path_for_init)
            self.bert = AutoModelForMaskedLM.from_config(config=l_config)
            self.bert.cls = nn.Identity() 
            self.dropout = nn.Dropout(l_config.hidden_dropout_prob if hasattr(l_config, 'hidden_dropout_prob') else 0.1)
            self.classifier = nn.Linear(l_config.hidden_size, num_labels)
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.bert.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            pooled_output = outputs.hidden_states[-1][:,0,:] 
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

    model = MyFineTunedBert(config_path_for_init=args.config_path, num_labels=2)

    if args.finetuned_checkpoint:
        model.load_state_dict(torch.load(args.finetuned_checkpoint, map_location=DEVICE), strict=False)
        print(f"Loaded fine-tuned model from {args.finetuned_checkpoint}")
    else:
        print("Error: No fine-tuned checkpoint specified. Prediction requires a fine-tuned model.")
        exit()

    model.to(DEVICE)
    model.eval()

    inputs = preprocess_url(args.url, tokenizer, max_len=MAX_PREDICT_LEN)
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)
    token_type_ids = inputs['token_type_ids'].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    labels = ['benign', 'malicious']
    print(f"URL: {args.url}")
    print(f"Predicted as: {labels[prediction]}")
    print(f"Probabilities: Benign={probabilities[0][0]:.4f}, Malicious={probabilities[0][1]:.4f}")