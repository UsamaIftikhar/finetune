from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load the trained model and tokenizer
model_name = "./trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Read the context from the data.txt file
with open('data.txt', 'r') as f:
    context = f.read().strip()

# Read the questions from the questions.txt file
with open('questions.txt', 'r') as f:
    questions = [line.strip() for line in f]

# Function to get answer from context using the trained model
def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    input_ids = inputs["input_ids"].tolist()[0]
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    
    # Get the most likely answer span
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    
    # Validate the answer span
    if answer_start >= len(input_ids) or answer_end >= len(input_ids) or answer_start > answer_end:
        return "Unable to find answer"
    
    # Extract the answer tokens from the input_ids
    answer_tokens = input_ids[answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    # Debugging information
    print(f"Question: {question}")
    print(f"Input IDs: {input_ids}")
    print(f"Answer Start Scores: {answer_start_scores}")
    print(f"Answer End Scores: {answer_end_scores}")
    print(f"Answer Start: {answer_start}")
    print(f"Answer End: {answer_end}")
    print(f"Answer Tokens: {answer_tokens}")
    print(f"Answer: {answer}")
    
    return answer

# Answer each question
for question in questions:
    answer = get_answer(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
