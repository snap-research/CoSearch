# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dual-Agent PPO Trainer for co-training Search-R1 and Reranker agents.

Training Strategy:
- Alternating epochs: Train Search-R1, then train Reranker
- Search-R1 training: Standard agent loop, reranker as deterministic tool
- Reranker training: Counterfactual rollout with GRPO grouping
"""

import uuid
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage, compute_response_mask, compute_reward
from verl.trainer.ppo.utils import Role
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


class DualAgentRayPPOTrainer(RayPPOTrainer):
    """PPO Trainer for co-training two agents (Search-R1 + Reranker).
    
    Training alternates between:
    1. Training Search-R1 (reranker fixed as deterministic tool)
    2. Training Reranker (with counterfactual rollout + GRPO)
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize dual-agent trainer."""
        super().__init__(*args, **kwargs)
        
        # Training schedule configuration
        self.search_r1_epochs_per_cycle = self.config.trainer.get("search_r1_epochs_per_cycle", 5)
        self.reranker_epochs_per_cycle = self.config.trainer.get("reranker_epochs_per_cycle", 1)
        self.total_epochs_per_cycle = self.search_r1_epochs_per_cycle + self.reranker_epochs_per_cycle
        
        # Separate dataloaders for each agent
        # For search-r1: normal Q&A data
        # For reranker: same data but processed differently during rollout
        self.search_r1_dataloader = self.train_dataloader
        self.reranker_dataloader = self.train_dataloader  # Same data, different processing
    
    def fit(self):
        """Main training loop with alternating epochs."""
        from verl.utils.tracking import Tracking
        
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
        
        # Progress bar
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress"
        )
        
        self.global_steps += 1
        
        # Main training loop
        for epoch in range(self.config.trainer.total_epochs):
            # Determine which agent to train
            epoch_in_cycle = epoch % self.total_epochs_per_cycle
            
            if epoch_in_cycle < self.search_r1_epochs_per_cycle:
                # Train Search-R1
                train_mode = "search_r1"
                dataloader = self.search_r1_dataloader
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}: Training Search-R1 Agent")
                print(f"{'='*60}")
            else:
                # Train Reranker
                train_mode = "reranker"
                dataloader = self.reranker_dataloader
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}: Training Reranker Agent (Counterfactual Rollout)")
                print(f"{'='*60}")
            
            # Run training for this epoch
            epoch_metrics = self._train_epoch(
                epoch=epoch,
                dataloader=dataloader,
                train_mode=train_mode,
                logger=logger,
                progress_bar=progress_bar,
            )
            
            # Log epoch summary
            logger.log(data=epoch_metrics, step=self.global_steps)
            
            # Validation
            if (self.val_reward_fn is not None and
                self.config.trainer.test_freq > 0 and
                (epoch + 1) % self.config.trainer.test_freq == 0):
                val_metrics = self._validate()
                logger.log(data=val_metrics, step=self.global_steps)
            
            # Checkpoint
            if self.config.trainer.save_freq > 0 and (epoch + 1) % self.config.trainer.save_freq == 0:
                self._save_checkpoint()
        
        progress_bar.close()
        print(f"\nTraining completed!")
    
    def _train_epoch(self, epoch, dataloader, train_mode, logger, progress_bar):
        """Train one epoch for the specified agent.
        
        Args:
            epoch: Current epoch number.
            dataloader: Dataloader for this epoch.
            train_mode: "search_r1" or "reranker".
            logger: Metrics logger.
            progress_bar: Progress bar.
            
        Returns:
            Dictionary of epoch-level metrics.
        """
        epoch_metrics = {
            "train_mode": train_mode,
            "epoch": epoch,
        }
        
        for batch_dict in dataloader:
            metrics = {}
            timing_raw = {}
            
            is_last_step = self.global_steps >= self.total_training_steps
            
            with marked_timer("step", timing_raw):
                # Prepare batch
                batch, gen_batch = self._prepare_generate_batch(batch_dict)
                
                # Add train_mode to meta_info
                gen_batch.meta_info["train_mode"] = train_mode
                
                # Generate sequences (different logic for each mode)
                with marked_timer("gen", timing_raw, color="red"):
                    gen_batch_output = self._generate_sequences(gen_batch, train_mode)
                    timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                
                # Post-process generation output
                batch = self._post_generate_batch(batch, gen_batch_output, metrics)
                
                # Compute rewards
                with marked_timer("reward", timing_raw, color="yellow"):
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                    batch.batch["token_level_scores"] = reward_tensor
                
                # Compute advantages
                with marked_timer("adv", timing_raw, color="brown"):
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        config=self.config.algorithm,
                    )
                
                # Update the appropriate agent
                if train_mode == "search_r1":
                    self._update_search_r1(batch, metrics, timing_raw)
                elif train_mode == "reranker":
                    self._update_reranker(batch, metrics, timing_raw)
            
            # Collect metrics
            metrics.update({
                "training/global_step": self.global_steps,
                "training/epoch": epoch,
                "training/train_mode": train_mode,
            })
            
            # Log
            logger.log(data=metrics, step=self.global_steps)
            progress_bar.update(1)
            self.global_steps += 1
            
            if is_last_step:
                break
        
        return epoch_metrics
    
    def _generate_sequences(self, gen_batch, train_mode):
        """Generate sequences based on training mode.
        
        Args:
            gen_batch: Generation batch.
            train_mode: "search_r1" or "reranker".
            
        Returns:
            Generated sequences.
        """
        # Call agent loop manager with appropriate mode
        # This assumes you've modified agent_loop_manager to support train_mode
        if hasattr(self.actor_rollout_wg, "generate_sequences_with_mode"):
            # If using extended agent loop manager
            return self.actor_rollout_wg.generate_sequences_with_mode(
                gen_batch,
                train_mode=train_mode
            )
        else:
            # Fallback to standard generation
            return self.actor_rollout_wg.generate_sequences(gen_batch)
    
    def _update_search_r1(self, batch, metrics, timing_raw):
        """Update Search-R1 agent (reranker fixed).
        
        Args:
            batch: Training batch.
            metrics: Metrics dictionary to update.
            timing_raw: Timing dictionary.
        """
        with marked_timer("update_search_r1", timing_raw, color="red"):
            # Standard actor update
            actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update({f"search_r1/{k}": v for k, v in actor_metrics.items()})
    
    def _update_reranker(self, batch, metrics, timing_raw):
        """Update Reranker agent (with GRPO on counterfactual branches).
        
        Args:
            batch: Training batch with GRPO grouping.
            metrics: Metrics dictionary to update.
            timing_raw: Timing dictionary.
        """
        with marked_timer("update_reranker", timing_raw, color="blue"):
            # Update reranker worker group
            # This assumes you have a separate reranker_wg
            if hasattr(self, "reranker_wg"):
                reranker_output = self.reranker_wg.update_actor(batch)
                reranker_metrics = reduce_metrics(reranker_output.meta_info["metrics"])
                metrics.update({f"reranker/{k}": v for k, v in reranker_metrics.items()})
            else:
                # If reranker shares same worker group, still update but with different data
                actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update({f"reranker/{k}": v for k, v in actor_metrics.items()})
            
            # Log GRPO-specific metrics
            if "num_groups" in batch.meta_info:
                metrics["reranker/num_grpo_groups"] = batch.meta_info["num_groups"]
                metrics["reranker/branches_per_group"] = batch.meta_info["num_branches_per_group"]
    
    def _prepare_generate_batch(self, batch_dict):
        """Prepare generation batch (same as parent class)."""
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        
        # Add uid
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))],
            dtype=object
        )
        
        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch = gen_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.n,
            interleave=True
        )
        
        return batch, gen_batch
    
    def _post_generate_batch(self, batch, gen_batch_output, metrics):
        """Post-process generation output (same as parent class)."""
        batch = batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.n,
            interleave=True
        )
        batch = batch.union(gen_batch_output)
        
        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        
        # Balance batch if configured
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)
        
        return batch
