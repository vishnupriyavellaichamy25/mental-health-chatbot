import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
import pandas as pd

class SafetyClassifier:
    def __init__(self, model_dir='models/crisis_classifier'):
        """
        Initializes the SafetyClassifier. Loads the fine-tuned model if it exists.
        """
        self.model_dir = model_dir
        self.model_name = "bert-base-uncased"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if os.path.exists(self.model_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = None

    def train(self):
        """
        Loads dataset, fine-tunes bert-base-uncased for 3 epochs,
        and saves to models/crisis_classifier.
        """
        print("Loading mental health crisis dataset...")
        # Since no specific dataset is mentioned, we create a mock synthetic one for this exercise
        # 0=normal, 1=distress, 2=crisis
        data = {
            'text': [
                "I am feeling okay today.", "Life is good.", "Just reading a book.",
                "I feel really sad and overwhelmed.", "I can't cope with this stress anymore.", "Everything is too much.",
                "I want to end it all.", "I have a plan to hurt myself.", "There is no point in living."
            ],
            'label': [0, 0, 0, 1, 1, 1, 2, 2, 2]
        }
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)
            
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3).to(self.device)
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=10,
            save_total_limit=2,
            logging_dir='./logs',
            report_to="none" # Disable wandb reporting
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets,
        )
        
        print("Training model for 3 epochs...")
        trainer.train()
        
        print(f"Saving model to {self.model_dir}...")
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

    def classify(self, text: str) -> int:
        """
        Classifies the text and returns the label (0=normal, 1=distress, 2=crisis).
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained. Call train() first or ensure models/crisis_classifier exists.")
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            label = torch.argmax(predictions, dim=1).item()
            
        return label

# Global instances for the respond function
classifier_instance = None

def safe_respond(user_message: str, chat_history: list, rag_pipeline=None) -> str:
    """
    Wraps the RAG pipeline response and appends crisis helplines if label >= 1.
    """
    global classifier_instance
    if classifier_instance is None:
        classifier_instance = SafetyClassifier()
        if not os.path.exists('models/crisis_classifier'):
            classifier_instance.train()
            
    label = classifier_instance.classify(user_message)
    
    # Generate base response using RAG pipeline
    if rag_pipeline:
        response = rag_pipeline.rag_respond(user_message, chat_history)
    else:
        response = "I am here to listen."

    if label >= 1:
        append_text = "\\n\\n***\\n**Crisis Helpline Resources:**\\nIf you are feeling distressed or in a crisis, please reach out for professional help immediately.\\n- iCall India: 9152987821\\n- Vandrevala: 1860-2662-345"
        response += append_text
        
    return response

if __name__ == "__main__":
    clf = SafetyClassifier()
    clf.train()
