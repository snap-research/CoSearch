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
Reranker Reward Loop Manager Stub

This is a minimal stub implementation required for VERL v0.7.0+ compatibility.
When reward_model.enable=False, VERL still requires a RewardLoopManager to be
registered even though it's never used.

The actual reward computation happens in RewardManager (old system) called from
ray_trainer.compute_reward(). This stub exists only to satisfy the registration
check in RewardManagerWorker.__init__().
"""

from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase


@register("reranker")
class RerankerRewardLoopManager(RewardLoopManagerBase):
    """
    Stub RewardLoopManager for reranker training.
    
    This class is registered but never instantiated when using:
    - reward_model.enable=False
    - Batch reward computation in ray_trainer
    
    If this class is ever instantiated, it indicates a configuration error.
    """

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        # Don't raise error here, just pass through
        # Only raise if run_single is actually called

    async def run_single(self, data):
        """
        Stub method for single-item async reward computation.
        
        This should never be called because:
        1. reward_model.enable=False disables RewardManagerWorker
        2. agent_loop async reward is disabled via commented logic
        3. All rewards computed via RewardManager.__call__() in ray_trainer
        
        If called, it indicates the async reward path is still active.
        """
        raise RuntimeError(
            "RerankerRewardLoopManager.run_single() should never be called. "
            "This indicates agent_loop async reward is still enabled. "
            "Please verify: "
            "1) reward_model.enable=False in config "
            "2) agent_loop.py line ~527: 'or not self.config.reward_model.enable' is commented out "
            "3) Only ray_trainer.compute_reward() is being used for reward computation"
        )
