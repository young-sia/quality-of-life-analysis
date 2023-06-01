import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification

# Step 2: Load KoBERT model and tokenizer
model_name = "skt/kobert-base-v1"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 for positive, negative, neutral

# Step 3: Prepare your dataset (X: input text, y: corresponding sentiment labels)
train_dataset = ...  # Prepare your training dataset
test_dataset = ...  # Prepare your testing dataset


# Step 4: Tokenize and encode the text
def tokenize_text(text):
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,  # Adjust as per your requirements
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return tokens["input_ids"], tokens["attention_mask"]

# Step 5: Create data loaders
batch_size = 16  # Adjust as per your requirements
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 6: Fine-tune KoBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 7: Train the model
optimizer = AdamW(model.parameters(), lr=2e-5)  # Adjust learning rate as per your requirements
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 3  # Adjust as per your requirements

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids, attention_mask = tokenize_text(batch["text"])
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping if needed

        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss}")

# Step 8: Evaluate the model
model.eval()
eval_loss, eval_accuracy = 0, 0

for batch in test_dataloader:
    input_ids, attention_mask = tokenize_text(batch["text"])
    labels = batch["labels"].to(device)

    with torch.no_grad():
        outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels)

    logits = outputs.logits
    _, preds = torch.max(logits, dim=1)

    loss = outputs.loss
    eval_loss += loss.item()
    eval_accuracy += torch.sum(preds == labels).item()

avg_loss = eval_loss / len(test_dataloader)
accuracy = eval_accuracy / len(test_dataset)
print(f"Test Loss: {avg_loss} - Accuracy: {accuracy}")


# Step 9: Perform sentiment analysis on new text
def perform_sentiment_analysis(text):
    model.eval()
    input_ids, attention_mask = tokenize_text(text)

    with torch.no_grad():
        outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device))

    logits = outputs.logits
    _, predicted_label = torch.max(logits, dim=1)

    return predicted_label.item()





