"""
Reranker Reward Manager for VERL Training

This module provides a reward manager specifically designed for reranker training.
It integrates with VERL's reward system and computes ranking-based rewards
(MRR, NDCG, Recall) with format validation.

Key Features:
1. Validates reranker output format (reason, select, rerank blocks)
2. Computes ranking metrics using pytrec_eval
3. Applies format penalty for invalid outputs
4. Supports multiple datasets (MS MARCO, TREC DL)
5. Integrates with VERL's DataProto structure
"""

import json
import torch
from typing import List, Optional

from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager import register


@register("reranker")
class RerankerRewardManager(AbstractRewardManager):
    """
    Reward manager for reranker training.
    
    This class computes rewards based on ranking quality metrics (MRR, NDCG)
    and applies format penalties for invalid outputs.
    
    Reward Formula:
        reward = format_penalty (if invalid) OR ranking_score (if valid)
        
    where:
        - format_penalty: -0.2 (default) for format errors
        - ranking_score: 0.0 to 1.0 based on MRR@M or NDCG@M
    
    Expected data format (in DataProto.batch['reward_model']):
        {
            'ground_truth': {
                'model_type': str,  # e.g., "qwen2507_distilled_instruct_thinking_rerank"
                'docids': List[str],  # Document IDs
                'qrels': Dict[str, int],  # Relevance judgments
                'N': int,  # Total candidates
                'M': int,  # Top-M to select
                'binary_qrels': Optional[Dict[str, int]],  # For TREC recall
            },
            'data_source': str,  # e.g., "msmarco_dev", "trec19"
        }
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score=None,
        reward_fn_key: str = "data_source",
        **kwargs,
    ):
        """
        Initialize the reranker reward manager.
        
        Args:
            tokenizer: Tokenizer for decoding model outputs
            num_examine: Number of examples to print for debugging (0 = none)
            compute_score: Reward computation function with signature:
                compute_score(data_source, solution_str, ground_truth, extra_info) -> float or dict
                If None, will use a default function (not recommended for reranker)
            reward_fn_key: Key to access data_source from non_tensor_batch (default: "data_source")
            **kwargs: Additional arguments (stored but not used by RerankerRewardManager itself)
        
        Note:
            The compute_score function should be provided via VERL's custom_reward_function config:
            
            custom_reward_function:
                path: /path/to/reranker_reward_functions.py
                name: compute_mrr_penalty_score  # or other function name
                reward_kwargs:
                    format_penalty: -0.2
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.examine_count = 0
        
        if compute_score is None:
            raise ValueError(
                "compute_score must be provided for RerankerRewardManager. "
                "Please configure custom_reward_function in your config:\n"
                "  custom_reward_function:\n"
                "    path: /path/to/reranker_reward_functions.py\n"
                "    name: compute_mrr_penalty_score\n"
                "    reward_kwargs:\n"
                "      format_penalty: -0.2"
            )
        
        self.compute_score = compute_score
        
        print(f"RerankerRewardManager initialized:")
        print(f"  - reward_fn_key: {reward_fn_key}")
        print(f"  - num_examine: {num_examine}")
        print(f"  - compute_score: {compute_score}")
        if kwargs:
            print(f"  - extra_kwargs: {kwargs}")

    def __call__(self, data, return_dict: bool = False):
        """
        Compute rewards for a batch of rollout data.
        
        Args:
            data: DataProto object with batch data
            return_dict: If True, return dict with rewards and extra info
        
        Returns:
            reward_tensor: Tensor of shape [batch_size, seq_length] with reward at last valid token
            OR dict with 'reward_tensor' and 'reward_extra_info' if return_dict=True
        """
        from collections import defaultdict
        
        # Initialize reward tensor with zeros, shape: [batch_size, seq_length]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        already_print_data_sources = {}
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            # Extract prompt and response following NaiveRewardManager's approach
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # Extract reward model data
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            
            # Parse qrels if stored as JSON string
            if isinstance(ground_truth['qrels'], str):
                ground_truth['qrels'] = json.loads(ground_truth['qrels'])
            
            if ground_truth.get('binary_qrels') is not None and isinstance(ground_truth['binary_qrels'], str):
                ground_truth['binary_qrels'] = json.loads(ground_truth['binary_qrels'])
            
            # Build extra_info dict
            extra_info = {
                'model_type': ground_truth['model_type'],
                'docids': ground_truth['docids'],
                'qrels': ground_truth['qrels'],
                'N': ground_truth['N'],
                'M': ground_truth['M'],
            }
            
            if ground_truth.get('binary_qrels') is not None:
                extra_info['binary_qrels'] = ground_truth['binary_qrels']
            
            # Compute score using the custom reward function
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth['qrels'],
                extra_info=extra_info,
            )
            
            # Handle dict or scalar score
            if isinstance(score, dict):
                reward = score['score']
                # Store extra information
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            
            # CRITICAL: Set reward at the last valid response token position
            reward_tensor[i, valid_response_length - 1] = reward
            
            # Print examples for debugging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                self._print_example(i, prompt_str, response_str, reward, data_source, ground_truth, score)
        
        if return_dict:
            return {
                'reward_tensor': reward_tensor,
                'reward_extra_info': reward_extra_info,
            }
        
        return reward_tensor

    def _print_example(self, idx, prompt_str, response_str, reward, data_source, ground_truth, score):
        """Print a single example for debugging (following NaiveRewardManager style)."""
        print("\n" + "=" * 80)
        print(f"[REWARD EXAMPLE {idx + 1}]")
        print("=" * 80)
        print(f"[data_source] {data_source}")
        print(f"[prompt] {prompt_str[:200]}{'...' if len(prompt_str) > 200 else ''}")
        print(f"[response] {response_str[:500]}{'...' if len(response_str) > 500 else ''}")
        
        # Print qrels
        qrels = ground_truth['qrels']
        if isinstance(qrels, str):
            qrels = json.loads(qrels)
        print(f"[ground_truth] {qrels}")
        
        # Print score details if dict
        if isinstance(score, dict):
            for key, value in score.items():
                print(f"[{key}] {value}")
        else:
            print(f"[score] {score}")
        
        print(f"[reward] {reward}")
        print(f"[N] {ground_truth['N']}, [M] {ground_truth['M']}")
        print("=" * 80)


if __name__ == "__main__":
    # Test the reward manager
    print("Testing RerankerRewardManager...")
    
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # Create reward manager
    reward_manager = RerankerRewardManager(
        tokenizer=tokenizer,
        num_examine=2,
        format_penalty=-0.2,
    )
    
    print("\n✓ RerankerRewardManager initialized successfully")
    print("\nTo test with actual data, run:")
    print("  python main_rerank_grpo.py --config-name reranker_grpo")
