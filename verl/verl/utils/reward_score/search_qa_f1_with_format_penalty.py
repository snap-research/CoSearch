# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string
from collections import Counter


def validate_response_structure(processed_str: str, do_print: bool, answer_turn=False) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")

    # Define required tags
    if answer_turn:
        tags = {
            'think_start': ('<reason>', 1),
            'think_end': ('</reason>', 1),
            'answer_start': ('<answer>', 1),
            'answer_end': ('</answer>', 1)
        }
    else:
        tags = {
            'think_start': ('<reason>', 1),
            'think_end': ('</reason>', 1),
            'answer_start': ('<tool_call>', 1),
            'answer_end': ('</tool_call>', 1)
        }

    # Step 1: Check if all required tags exist with correct count
    positions = {}
    all_tags_present = True
    
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        pos = processed_str.find(tag_str)
        positions[tag_name] = pos
        
        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            if do_print:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            all_tags_present = False
    
    if not all_tags_present:
        if do_print:
            print("  [Failed] Not all required tags present")
        return False
    
    # Step 2: Verify tag order (now we know all tags exist)
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <reason>...</reason><tool_call/answer>...</tool_call/answer>")
        return False
    else:
        if do_print:
            print("  Tag sequence validation passed")

    # Step 3: Check if <reason> content is not empty
    think_start_idx = positions['think_start'] + len('<reason>')
    think_end_idx = positions['think_end']
    
    if think_start_idx >= think_end_idx:
        if do_print:
            print(f"  [Error] <reason> tags are malformed (start={think_start_idx}, end={think_end_idx})")
        return False
    
    think_content = processed_str[think_start_idx:think_end_idx].strip()
    if not think_content:
        if do_print:
            print("  [Error] <reason>...</reason> is empty or only contains whitespace")
        return False
    else:
        if do_print:
            print(f"  <reason>...</reason> is valid (length={len(think_content)})")

    # Step 4: Check if answer/tool_call content is not empty
    if answer_turn:
        answer_start_idx = positions['answer_start'] + len('<answer>')
        answer_end_idx = positions['answer_end']
        tag_name = '<answer>'
    else:
        answer_start_idx = positions['answer_start'] + len('<tool_call>')
        answer_end_idx = positions['answer_end']
        tag_name = '<tool_call>'
    
    if answer_start_idx >= answer_end_idx:
        if do_print:
            print(f"  [Error] {tag_name} tags are malformed (start={answer_start_idx}, end={answer_end_idx})")
        return False
    
    answer_content = processed_str[answer_start_idx:answer_end_idx].strip()
    if not answer_content:
        if do_print:
            print(f"  [Error] {tag_name}...{tag_name.replace('<', '</')} is empty or only contains whitespace")
        return False
    else:
        if do_print:
            print(f"  {tag_name}...{tag_name.replace('<', '</')} is valid (length={len(answer_content)})")

    return True

def compute_format_reward(full_text: str) -> list[float]:
    # Extract all assistant responses
    assistant_blocks = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", full_text, re.DOTALL)
    
    format_rewards = []
    for i, block in enumerate(assistant_blocks): 
        if i == len(assistant_blocks) - 1: 
            format_r = validate_response_structure(block, do_print=False, answer_turn=True)
        else:
            format_r = validate_response_structure(block, do_print=False, answer_turn=False) 

        format_rewards.append(format_r) 

    return all(format_rewards)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    f1_score = 0. 
    for golden_answer in golden_answers:
        f1_score = max(f1_score, f1(prediction, golden_answer))
    return f1_score

def f1(prediction, answer):
    """Compute the F1 score between the prediction and the answer.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(answer).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def search_qa_f1_penalty_compute_score(data_source, solution_str, ground_truth, extra_info, format_penalty=-0.2, **kwargs):
    """The scoring function for F1.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        extra_info: extra information
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_penalty: the penalty for incorrect format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    assert format_penalty <= 0.0, "format_penalty should be non-positive"

    ans_score = 0.0 
    if answer is not None:
        ans_score = compute_f1(answer, ground_truth["target"])

    is_format_correct = compute_format_reward(solution_str) 

    if is_format_correct:
        total_score = ans_score
    else:
        total_score = format_penalty
    
    result = {
        "score": total_score,
        "valid": is_format_correct,
        "f1": ans_score,
    }
    if extra_info["json_correct"] is not None:
        result["json_correct"] = extra_info["json_correct"]

    if extra_info["one_tool_call_per_assistant"] is not None:
        result["one_tool_call_per_assistant"] = extra_info["one_tool_call_per_assistant"]

    # print(f"🔧 [DEBUG] answer: {answer}, ground_truth: {ground_truth['target']}, score: {total_score}, ans_score: {ans_score}, format_correct: {float(is_format_correct)}")
    return result