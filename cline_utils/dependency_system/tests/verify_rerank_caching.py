
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from cline_utils.dependency_system.analysis import embedding_manager
from cline_utils.dependency_system.utils.cache_manager import cache_manager

class TestRerankCaching(unittest.TestCase):
    def setUp(self):
        # Clear caches before each test
        cache_manager.clear_all()
        
    @patch('cline_utils.dependency_system.analysis.embedding_manager._load_reranker_model')
    @patch('cline_utils.dependency_system.analysis.embedding_manager._get_available_vram')
    @patch('cline_utils.dependency_system.analysis.embedding_manager._get_available_ram')
    def test_reranking_cache(self, mock_get_ram, mock_get_vram, mock_load_model):
        # Setup mocks
        mock_get_ram.return_value = 16.0
        mock_get_vram.return_value = 0.0 # Force CPU path or just ensure no CUDA errors
        
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Mock tokenizer behavior
        mock_tokenizer.pad.return_value = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda x: 1 if x == 'no' else (2 if x == 'yes' else 0)
        mock_tokenizer.encode.return_value = [1, 2, 3]
        
        # Mock the tokenizer call itself (batch encoding)
        # It should return a dict with 'input_ids' which is a list of lists
        def tokenizer_side_effect(texts, **kwargs):
            return {'input_ids': [[1, 2, 3]] * len(texts)}
        mock_tokenizer.side_effect = tokenizer_side_effect
        
        # Mock model behavior
        # Return logits that will result in some score
        # Shape: [batch_size, seq_len, vocab_size]
        # We need vocab_size >= 3 to cover ids 0, 1, 2
        mock_logits = torch.randn(1, 3, 10) 
        mock_model.return_value.logits = mock_logits
        # Use side_effect to return a NEW iterator each time
        mock_model.parameters.side_effect = lambda: iter([MagicMock(device=MagicMock(type='cpu'))])

        mock_load_model.return_value = (mock_tokenizer, mock_model)
        
        # Inject pre-computed token IDs since we are mocking the load function which usually sets them
        embedding_manager.reranker_false_id = 1
        embedding_manager.reranker_true_id = 2
        
        query = "test query"
        candidates = ["doc1", "doc2"]
        
        print("First call to rerank...")
        # First call
        results1 = embedding_manager.rerank_candidates_with_qwen3(query, candidates)
        
        # Check cache stats
        cache = cache_manager.get_cache("reranking")
        print(f"Cache hits after first call: {cache.metrics.hits}")
        print(f"Cache misses after first call: {cache.metrics.misses}")
        
        self.assertEqual(cache.metrics.hits, 0)
        self.assertEqual(cache.metrics.misses, 1)
        
        print("Second call to rerank (same inputs)...")
        # Second call - should hit cache
        results2 = embedding_manager.rerank_candidates_with_qwen3(query, candidates)
        
        print(f"Cache hits after second call: {cache.metrics.hits}")
        print(f"Cache misses after second call: {cache.metrics.misses}")
        
        self.assertEqual(cache.metrics.hits, 1)
        self.assertEqual(cache.metrics.misses, 1)
        self.assertEqual(results1, results2)
        
        print("Third call to rerank (different inputs)...")
        # Third call - different inputs
        results3 = embedding_manager.rerank_candidates_with_qwen3(query, ["doc3"])
        
        print(f"Cache hits after third call: {cache.metrics.hits}")
        print(f"Cache misses after third call: {cache.metrics.misses}")
        
        self.assertEqual(cache.metrics.hits, 1)
        self.assertEqual(cache.metrics.misses, 2)

if __name__ == '__main__':
    unittest.main()
