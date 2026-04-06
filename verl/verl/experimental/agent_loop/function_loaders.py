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
Utility functions for loading configurable functions from config.

This module provides generic function loading utilities similar to VERL's
reward function loading mechanism, but applicable to any configurable functions.
"""

import importlib.util
import inspect
import os
import sys
from functools import partial
from typing import Callable, Optional

from omegaconf import DictConfig


def load_custom_function(
    config: DictConfig,
    config_key: str,
    kwargs_key: str = None,
    default_fn: Optional[Callable] = None,
) -> Optional[Callable]:
    """Load a custom function from external file based on config.
    
    This follows the same pattern as VERL's reward function loading:
    1. Read path and function name from config
    2. Dynamically import the Python module
    3. Get the function object
    4. Wrap with functools.partial to inject kwargs
    
    Args:
        config: OmegaConf DictConfig containing configuration.
        config_key: Key in config for the function config (e.g., "reranker_uid_group_function").
        kwargs_key: Key for the kwargs dict (e.g., "uid_group_kwargs"). 
                    If None, defaults to "{config_key}_kwargs".
        default_fn: Default function to return if config is not specified.
    
    Returns:
        Callable function, wrapped with functools.partial if kwargs are provided.
        Returns default_fn if config is not specified.
    
    Example:
        In config:
            reranker_uid_group_function:
                path: "/path/to/uid_group_functions.py"
                name: "uid_group_basic"
                uid_group_kwargs:
                    threshold: 0.8
        
        Usage:
            uid_group_fn = load_custom_function(
                config, 
                "reranker_uid_group_function",
                "uid_group_kwargs",
                default_fn=uid_group_basic
            )
    """
    # Get function config from main config
    fn_config = config.get(config_key)
    
    # If config is not specified or path is null, return default
    if fn_config is None or fn_config.get("path") is None:
        return default_fn
    
    file_path = fn_config.get("path")
    function_name = fn_config.get("name")
    
    # Determine kwargs key
    if kwargs_key is None:
        # Extract base name from config_key (e.g., "reranker_uid_group_function" -> "uid_group")
        # Default pattern: remove "reranker_" prefix and "_function" suffix, add "_kwargs"
        kwargs_key = config_key.replace("reranker_", "").replace("_function", "") + "_kwargs"
    
    fn_kwargs = dict(fn_config.get(kwargs_key, {}))
    
    # Dynamically load the module
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Custom function file not found: {file_path}")
    
    # Import module from file path
    spec = importlib.util.spec_from_file_location("custom_function_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_function_module"] = module
    spec.loader.exec_module(module)
    
    # Get the function from module
    if not hasattr(module, function_name):
        raise AttributeError(f"Function {function_name} not found in {file_path}")
    
    raw_fn = getattr(module, function_name)
    
    # Verify it's callable
    if not callable(raw_fn):
        raise TypeError(f"{function_name} in {file_path} is not callable")
    
    # Wrap with functools.partial to inject kwargs
    if fn_kwargs:
        # Check if function is async
        if inspect.iscoroutinefunction(raw_fn):
            return partial(_call_with_kwargs_async, raw_fn, fn_kwargs)
        else:
            return partial(_call_with_kwargs, raw_fn, fn_kwargs)
    else:
        return raw_fn


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls raw_fn by merging extra_kwargs into call-time kwargs.
    
    Extra kwargs take precedence (can override call-time kwargs).
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
    """Async version of _call_with_kwargs."""
    merged_kwargs = {**kwargs, **extra_kwargs}
    return await raw_fn(*args, **merged_kwargs)
