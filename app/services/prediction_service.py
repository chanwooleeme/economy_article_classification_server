import torch
from typing import List
from core.model_loader import get_tokenizer_pool, get_model

def predict_articles(texts: List[str]):
    with torch.no_grad():
        with get_tokenizer_pool().use() as tokenizer:
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            outputs = get_model()(**inputs, output_hidden_states=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1).tolist()
        hidden_states = outputs.hidden_states[-1]

        # Mean pooling
        input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).tolist()

    return predictions, probs, embeddings
