import json
from collections import Counter

# === MANUAL CONFIGURATION ===
train_json_path = "/home/mukul36/Documents/REWI1/pd_wi_hw5_word/train.json"
output_vocab_path = "/home/mukul36/Documents/REWI1/pd_wi_hw5_word/token_vocab.json"
fold_index = 0  # Cross-validation fold index (e.g., 0, 1, ...)
n = 2  # n-gram size (e.g., 2 for bigrams)

# === TOKENIZE FUNCTION ===
def tokenize(text, n=2):
    text = text.strip()
    return [text[i:i + n] for i in range(len(text) - n + 1)] if len(text) >= n else []

# === LOAD TRAINING TEXTS FROM ANNOTATIONS ===
with open(train_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

annotations = data["annotations"][str(fold_index)]
texts = [entry["label"] for entry in annotations]

# === COUNT TOKENS ===
token_counter = Counter()
for text in texts:
    tokens = tokenize(text, n=n)
    token_counter.update(tokens)

# === BUILD VOCABULARY ===
special_tokens = ["<BLANK>", "<UNK>"]
ordered_tokens = special_tokens + [token for token, _ in token_counter.most_common()]
token_to_index = {token: idx for idx, token in enumerate(ordered_tokens)}

# === SAVE VOCAB TO FILE ===
with open(output_vocab_path, 'w', encoding='utf-8') as f:
    json.dump(token_to_index, f, ensure_ascii=False, indent=4)

# === LOG INFO ===
print(f"✅ Saved vocabulary to: {output_vocab_path}")
print(f"🔢 Vocab size: {len(token_to_index)} tokens")
print(f"🔤 Top 10 tokens: {ordered_tokens[:10]}")
