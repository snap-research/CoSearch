"""
Reranker Datasets for Generation and Training

This module provides two PyTorch Dataset classes:
1. RerankerDataset: Simple dataset for generation/inference
2. RerankerRLHFDataset: RLHF dataset for GRPO/PPO training with VERL

Expected parquet columns:
    - prompt: List of chat messages (list of dicts with 'role' and 'content')
    - data_source: Dataset name (e.g., "msmarco_dev", "trec19")
    - reward_model: Dict with ground truth information
        - ground_truth:
            - model_type: Model type string
            - docids: List of document IDs
            - qrels: Relevance judgments dict
            - N: Number of candidates
            - M: Number to select
            - binary_qrels: (Optional) Binary qrels for TREC
"""

import json
import pandas as pd
import sys
from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset

# Add verl to path
verl_path = Path(__file__).parent.parent / "verl"
sys.path.insert(0, str(verl_path))

from verl.utils.dataset.rl_dataset import RLHFDataset


class RerankerDataset(Dataset):
    """
    Dataset for reranker generation/inference.
    
    This is a simple PyTorch Dataset that loads parquet files
    and returns prompts without RLHF-specific processing.
    
    Used by: main_rerank_generation.py
    """

    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        processor,
        config,
        max_samples: int = -1,
    ):
        """
        Initialize the reranker dataset.
        
        Args:
            data_files: List of paths to parquet files
            tokenizer: Tokenizer instance (not used in this simple dataset)
            processor: Processor instance (not used in this simple dataset)
            config: Data configuration object
            max_samples: Maximum number of samples to load (-1 for all)
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        # Load data files
        if isinstance(data_files, str):
            data_files = [data_files]
        
        print(f"Loading {len(data_files)} data file(s)...")
        datasets = []
        for data_file in data_files:
            df = pd.read_parquet(data_file)
            datasets.append(df)
            print(f"  ✓ Loaded {len(df)} examples from {data_file}")
        
        # Concatenate all datasets
        self.data = pd.concat(datasets, axis=0, ignore_index=True)
        
        # Apply max_samples
        if max_samples > 0 and len(self.data) > max_samples:
            self.data = self.data.iloc[:max_samples]
            print(f"  → Truncated to {max_samples} examples")
        
        print(f"Total dataset size: {len(self.data)}")
        
        # Get column keys from config
        self.prompt_key = config.get("prompt_key", "prompt")
        self.data_source_key = config.get("data_source_key", "data_source")
        self.reward_model_key = config.get("reward_model_key", "reward_model")
        
        # Validate required columns
        required_cols = [self.prompt_key, self.data_source_key, self.reward_model_key]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single example.
        
        Returns:
            Dict with keys:
                - data_source: str
                - prompt: List of chat messages
                - reward_model: Dict with ground truth information
        """
        row = self.data.iloc[idx]
        
        # Extract prompt (list of chat messages)
        prompt = row[self.prompt_key]
        if isinstance(prompt, str):
            # If stored as JSON string, parse it
            prompt = json.loads(prompt)
        
        # Convert to list if it's a numpy array
        if hasattr(prompt, 'tolist'):
            prompt = prompt.tolist()
        
        # Extract data source
        data_source = row[self.data_source_key]
        
        # Extract reward model data
        reward_model = row[self.reward_model_key]
        if isinstance(reward_model, str):
            reward_model = json.loads(reward_model)
        
        # Ensure ground_truth exists
        if 'ground_truth' not in reward_model:
            raise ValueError(f"Missing 'ground_truth' in reward_model for example {idx}")
        
        return {
            'data_source': data_source,
            'prompt': prompt,
            'reward_model': reward_model,
        }


class RerankerRLHFDataset(RLHFDataset):
    """
    Dataset for reranker GRPO/PPO training with VERL.
    
    Inherits from RLHFDataset to leverage automatic data processing:
    1. Parent loads parquet files
    2. Parent extracts messages from prompt_key
    3. Parent calls apply_chat_template() to convert messages → text
    4. Parent tokenizes text → input_ids, attention_mask
    5. Parent saves original messages to 'raw_prompt' for async rollout
    6. Child (this class) adds reranker-specific fields: data_source, reward_model
    
    Data Flow:
        Parquet → RLHFDataset.__init__ → loads self.dataset
        → RLHFDataset.__getitem__ → applies chat template + tokenization
        → RerankerRLHFDataset.__getitem__ → adds data_source, reward_model
        → collate_fn → creates batch with 'prompts', 'responses', 'raw_prompt', etc.
        → AgentLoop receives via kwargs["raw_prompt"] → regenerates tokens
        → vLLM generates → RewardManager computes rewards
    
    Used by: main_rerank_grpo.py
    """

    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        processor,
        config,
        max_samples: int = -1,
    ):
        """
        Initialize the reranker RLHF dataset.
        
        Args:
            data_files: List of paths to parquet files
            tokenizer: Tokenizer instance
            processor: Processor instance (can be None)
            config: Data configuration object with keys:
                - prompt_key: Column name for prompt (default: "prompt")
                - data_source_key: Column name for data source (default: "data_source")
                - reward_model_key: Column name for reward info (default: "reward_model")
                - max_prompt_length: Max prompt length (default: 1024)
                - truncation: How to handle long prompts (default: "error")
                - return_raw_chat: Whether to save raw messages (default: True for async)
                - apply_chat_template_kwargs: Additional kwargs for chat template
            max_samples: Maximum number of samples to load (-1 for all)
        """
        # Convert to list if needed
        if isinstance(data_files, str):
            data_files = [data_files]
        
        # Initialize parent RLHFDataset
        # This will:
        # 1. Download/cache parquet files
        # 2. Load into HuggingFace datasets.Dataset
        # 3. Apply chat template and tokenization
        # 4. Filter overlong prompts
        # 5. Save raw_prompt for async mode
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
            max_samples=max_samples,
        )
        
        # Store keys for additional reranker-specific fields
        self.data_source_key = config.get("data_source_key", "data_source")
        self.reward_model_key = config.get("reward_model_key", "reward_model")
        
        print(f"✓ RerankerRLHFDataset initialized with {len(self)} examples")

    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Returns dict from parent RLHFDataset.__getitem__() with additional keys:
            - data_source: str (dataset name)
            - reward_model: Dict with ground truth information
            
        Parent class automatically provides:
            - input_ids: torch.Tensor [prompt_length] - Tokenized prompt
            - attention_mask: torch.Tensor [prompt_length] - Attention mask
            - raw_prompt: List[Dict] - Original messages (for async rollout)
            - full_prompts: str - Text form of prompt (if return_full_prompt=True)
            
        The 'raw_prompt' field is crucial for async rollout:
            Dataset saves messages → collate_fn puts in non_tensor_batch["raw_prompt"]
            → AgentLoop receives via kwargs["raw_prompt"]
            → SingleTurnAgentLoop calls apply_chat_template() again
            → vLLM generates with fresh tokens
        """
        # Call parent's __getitem__ to get tokenized data
        # Parent will handle: messages → apply_chat_template → tokenization → raw_prompt saving
        item = super().__getitem__(idx)
        
        # Add reranker-specific fields from the raw dataset
        # Access parent's internal dataframe (HuggingFace Dataset object)
        row_dict = self.dataframe[idx]
        
        # Add data_source if present
        if self.data_source_key in row_dict:
            item['data_source'] = row_dict[self.data_source_key]
        
        # Add reward_model if present
        if self.reward_model_key in row_dict:
            reward_model = row_dict[self.reward_model_key]
            # Handle JSON string if stored as string in parquet
            if isinstance(reward_model, str):
                reward_model = json.loads(reward_model)
            item['reward_model'] = reward_model
        
        return item


if __name__ == "__main__":
    # Test both datasets
    print("Testing Reranker Datasets...")
    
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    
    # Load tokenizer
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test file path
    test_file = "/path/to/test.parquet"
    
    # Test RerankerDataset (simple)
    print("\n" + "=" * 80)
    print("Testing RerankerDataset (for generation/inference)")
    print("=" * 80)
    
    config_simple = OmegaConf.create({
        'prompt_key': 'prompt',
        'data_source_key': 'data_source',
        'reward_model_key': 'reward_model',
    })
    
    try:
        dataset_simple = RerankerDataset(
            data_files=[test_file],
            tokenizer=tokenizer,
            processor=None,
            config=config_simple,
            max_samples=10,
        )
        
        print(f"\n✓ RerankerDataset loaded: {len(dataset_simple)} examples")
        
        example = dataset_simple[0]
        print(f"\nFirst example keys: {example.keys()}")
        print(f"  data_source: {example.get('data_source', 'N/A')}")
        print(f"  prompt type: {type(example.get('prompt', 'N/A'))}")
        print(f"  reward_model keys: {example.get('reward_model', {}).keys() if 'reward_model' in example else 'N/A'}")
        
    except FileNotFoundError:
        print(f"\n⚠ Test file not found: {test_file}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Test RerankerRLHFDataset (RLHF)
    print("\n" + "=" * 80)
    print("Testing RerankerRLHFDataset (for GRPO/PPO training)")
    print("=" * 80)
    
    config_rlhf = OmegaConf.create({
        'prompt_key': 'prompt',
        'data_source_key': 'data_source',
        'reward_model_key': 'reward_model',
        'max_prompt_length': 1024,
        'truncation': 'error',
        'return_raw_chat': True,  # Essential for async rollout
        'filter_overlong_prompts': True,
        'apply_chat_template_kwargs': {},
    })
    
    try:
        dataset_rlhf = RerankerRLHFDataset(
            data_files=[test_file],
            tokenizer=tokenizer,
            processor=None,
            config=config_rlhf,
            max_samples=10,
        )
        
        print(f"\n✓ RerankerRLHFDataset loaded: {len(dataset_rlhf)} examples")
        
        example = dataset_rlhf[0]
        print(f"\nFirst example keys: {example.keys()}")
        print(f"  data_source: {example.get('data_source', 'N/A')}")
        print(f"  input_ids shape: {example['input_ids'].shape if 'input_ids' in example else 'N/A'}")
        print(f"  raw_prompt type: {type(example.get('raw_prompt', 'N/A'))}")
        print(f"  reward_model keys: {example.get('reward_model', {}).keys() if 'reward_model' in example else 'N/A'}")
        
    except FileNotFoundError:
        print(f"\n⚠ Test file not found: {test_file}")
        print("\nTo test with actual data, create a parquet file with the expected schema")
        print("\nExpected columns:")
        print("  - prompt: List of dicts [{'role': 'user', 'content': '...'}]")
        print("  - data_source: str (e.g., 'msmarco_dev')")
        print("  - reward_model: Dict with 'ground_truth' key")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

