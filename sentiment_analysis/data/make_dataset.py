from datasets import load_dataset

def load_and_process_dataset(dataset_name="rotten_tomatoes"):
    #init tokenizer 
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    # Load the dataset
    dataset = load_dataset(dataset_name, split="train")

    # Tokenize and pad the entire dataset
    tokenized_dataset = tokenizer(dataset['text'], padding=True, truncation=True, return_tensors='pt')

    # Convert labels to tensors
    labels = torch.tensor(dataset['label'])

    # Create DataLoader with aligned batch sizes
    dataloader = DataLoader(list(zip(tokenized_dataset['input_ids'], tokenized_dataset['attention_mask'], labels)), batch_size=batch_size, shuffle=True)

    return dataloader


if __name__ == '__main__':
 # Load and process the dataset
    train_dataloader = load_and_process_dataset()

    # Print some information about the dataset
    print(f"Number of training samples: {len(train_dataloader.dataset)}")
    print(f"Example input_ids: {train_dataloader.dataset[0][0]}")
    print(f"Example attention_mask: {train_dataloader.dataset[0][1]}")
    print(f"Example label: {train_dataloader.dataset[0][2]}")