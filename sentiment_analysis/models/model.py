from transformers import DistilBertForSequenceClassification

def load_pretrained_model():
    # Initialize and return the pre-trained model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return model

# Define any other functions related to your model architecture here
if __name__ == '__main__':
    # Load the pre-trained model
    model = load_pretrained_model()
    
    # Print the model architecture
    print("Pre-trained model loaded successfully! Here are the details: ")