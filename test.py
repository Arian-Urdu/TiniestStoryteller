import unittest
from unittest.mock import patch, MagicMock

import config as cfg
import generate

from transformers import PreTrainedTokenizerFast
import torch

class TestConfig(unittest.TestCase):

    def test_config_type(self):
        """Test that the config is a dictionary."""
        self.assertIsInstance(cfg.config, dict)

    def test_batch_size(self):
        """Test that batch_size is a positive integer."""
        self.assertIsInstance(cfg.batch_size, int)
        self.assertGreater(cfg.batch_size, 0)

    def test_curriculum_schedule(self):
        """Test that curriculum_schedule is correctly defined."""
        self.assertEqual(len(cfg.curriculum_schedule), 4)
        self.assertEqual(cfg.curriculum_schedule[-1], cfg.num_epochs)

    def test_tokenizer(self):
        """Test basic properties of the tokenizer."""
        self.assertIsInstance(cfg.tokenizer, PreTrainedTokenizerFast)
        self.assertEqual(cfg.tokenizer.bos_token, "<|endoftext|>")
        self.assertEqual(cfg.tokenizer.eos_token, "<|endoftext|>")
        self.assertEqual(cfg.tokenizer.pad_token, "[PAD]")
        self.assertEqual(cfg.vocab_size, len(cfg.tokenizer))



class TestGenerate(unittest.TestCase):

    def test_model_loading(self):
        """ Test that the model is loaded """
        self.assertIsNotNone(generate.model)

    def test_model_eval_mode(self):
        """ Test that the model is in evaluation mode """
        self.assertTrue(generate.model.training == False)

    @patch('torch.zeros')
    def test_context_creation(self, mock_zeros):
        """ Test the creation of the initial context """
        generate.num_gen = 1  # Ensure we only generate once
        with patch('builtins.print'):  # Suppress print output
            exec(open('generate.py').read())
        mock_zeros.assert_called_once_with((1, 1), dtype=torch.long, device=generate.device)

        
if __name__ == '__main__':
    unittest.main()