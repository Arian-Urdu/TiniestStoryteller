import unittest
from process_tinystories import load_and_prepare_data
from datasets import load_dataset
from transformers import GPT2TokenizerFast

class TestDataProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_data = load_and_prepare_data('TinyStoriesV2-GPT4-train.txt', num_samples=50)

    def test_data_loading(self):
        # Check if the dataset is properly loaded and tokenized
        self.assertTrue('input_ids' in self.sample_data.features)
        self.assertTrue('attention_mask' in self.sample_data.features)
        self.assertEqual(len(self.sample_data), 50, "Dataset should only contain 10 samples.")

    def test_token_structure(self):
        # Check if the tokens are correctly structured
        sample_entry = self.sample_data[0]
        self.assertIsInstance(sample_entry['input_ids'], list)
        self.assertIsInstance(sample_entry['attention_mask'], list)

    def test_tokenizer_length(self):
        # Ensure each tokenized output has correct length
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        for entry in self.sample_data:
            self.assertEqual(len(entry['input_ids']), 128)
            self.assertEqual(len(entry['attention_mask']), 128)

if __name__ == '__main__':
    unittest.main()