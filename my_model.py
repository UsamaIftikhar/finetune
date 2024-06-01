from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset

# Read the context from the data.txt file
with open('data.txt', 'r') as f:
    context = f.read().strip()

# Read the questions from the questions.txt file
with open('questions.txt', 'r') as f:
    questions = [line.strip() for line in f]

# Create the answers based on the context provided
answers = [
    {"text": "redhood tech", "answer_start": 7},
    {"text": "Pakistan", "answer_start": 41},
    {"text": "web development, mobile app development, and UI/UX work", "answer_start": 68},
    {"text": "1997", "answer_start": 54}
]

# Your custom dataset
data = {
    'id': [str(i) for i in range(1, len(questions) + 1)],
    'context': [context] * len(questions),
    'question': questions,
    'answers': answers
}

# Convert the data to a Dataset object
dataset = Dataset.from_dict(data)
train_test_split = dataset.train_test_split(test_size=0.5)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Tokenize data
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation=True,
        padding="max_length",
        return_overflowing_tokens=True,
        stride=128,
    )
    inputs["start_positions"] = [ans['answer_start'] for ans in examples["answers"]]
    inputs["end_positions"] = [ans['answer_start'] + len(ans['text']) for ans in examples["answers"]]
    return inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_test = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer, padding=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_steps=500,  # Evaluate every 500 steps
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,  # Limit the total checkpoints to 2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
