import datasets

def main():
    # Step 1: Load the dataset
    dataset = datasets.load_dataset("roneneldan/TinyStories")

    # Extract texts from both training and validation splits in one file
    train_texts = [example['text'] for example in dataset['train']]
    validation_texts = [example['text'] for example in dataset['validation']]
    all_texts = train_texts + validation_texts

    # Step 2: Write the text data to a file
    with open('data/tiny_stories_all.txt', 'w', encoding='utf-8') as f:
        for text in all_texts: f.write(text + '\n')

    print('saved dataset sucessfully in data/tiny_stories_all.txt')

if __name__ == '__main__':
    main()