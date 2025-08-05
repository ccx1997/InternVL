"""
Evaluation script for RoboSpatial-Home benchmark results.
"""

import os
import re
import ast
import json
from tqdm import tqdm
from datasets import load_dataset

def point_in_polygon(x, y, poly):
    """
    Check if the point (x, y) lies within the polygon defined by a list of (x, y) tuples.
    Uses the ray-casting algorithm.
    """
    num = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, num + 1):
        p2x, p2y = poly[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def evaluate_answer(ground_truth, generated_answer):
    """
    Evaluates if the generated answer is correct based on the ground truth.
    Returns a tuple of (is_correct, is_binary_answer, parsed_answer, is_parsable).
    """
    gen_answer = generated_answer.strip().lower()
    gt_lower = ground_truth.strip().lower()
    
    # Check if this is a binary yes/no question
    if gt_lower in ["yes", "no"]:
        is_binary = True
        is_gt_yes = (gt_lower == "yes")
        # Binary answers are always considered parsable if they contain text
        is_parsable = len(gen_answer) > 0
        if is_gt_yes:
            correct = gen_answer.startswith("yes")
        else:
            correct = gen_answer.startswith("no")
        return correct, is_binary, gen_answer, is_parsable
    else:
        # Numeric evaluation: ground_truth is a list of points defining a polygon
        is_binary = False
        parsed_answer = None
        is_parsable = False  # Default to not parsable until we successfully parse
        
        try:
            gt_polygon = ast.literal_eval(ground_truth)
            if not isinstance(gt_polygon, list) or len(gt_polygon) < 3:
                return False, is_binary, parsed_answer, is_parsable
            
            # Extract the first coordinate pair using regex
            # Look for patterns like (0.1,0.2) or (0.1, 0.2) or [0.1, 0.2] or [0.1,0.2]
            # This approach is more robust than trying to parse the entire list
            
            # Try to match tuple format (x,y) or (x, y)
            tuple_match = re.search(r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)', generated_answer)
            if tuple_match:
                try:
                    x = float(tuple_match.group(1))
                    y = float(tuple_match.group(2))
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except (ValueError, TypeError):
                    pass  # Continue to other formats if float conversion fails
            
            # Try to match list format [x,y] or [x, y]
            list_match = re.search(r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]', generated_answer)
            if list_match:
                try:
                    x = float(list_match.group(1))
                    y = float(list_match.group(2))
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except (ValueError, TypeError):
                    pass  # Continue to other formats if float conversion fails
            
            # Fall back to the original approach but with extra safety
            try:
                # Extract the first list (text between square brackets) from generated_answer
                # Use a regex that can handle multi-line content
                match = re.search(r'\[(.*?)\]', generated_answer, re.DOTALL)
                if match is None:
                    return False, is_binary, parsed_answer, is_parsable
                
                # Add spaces after commas if not present (to help ast.literal_eval)
                list_content = match.group(1)
                list_content = re.sub(r',(\S)', r', \1', list_content)
                
                # Try to fix truncated tuples by adding closing parenthesis and brackets if needed
                list_content = list_content.strip()
                if list_content.endswith(','):
                    list_content = list_content[:-1]
                
                list_str = '[' + list_content + ']'
                
                # Try to parse the list directly
                try:
                    gen_val = ast.literal_eval(list_str)
                except (SyntaxError, ValueError):
                    # If direct parsing fails, try to extract just the first tuple
                    tuple_match = re.search(r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)', list_content)
                    if tuple_match:
                        x = float(tuple_match.group(1))
                        y = float(tuple_match.group(2))
                        parsed_answer = (x, y)
                        is_parsable = True
                        correct = point_in_polygon(x, y, gt_polygon)
                        return correct, is_binary, parsed_answer, is_parsable
                    else:
                        return False, is_binary, parsed_answer, is_parsable
                
                # Handle different formats for points
                if isinstance(gen_val, list):
                    if len(gen_val) == 0:
                        return False, is_binary, parsed_answer, is_parsable
                        
                    # Case 1: The list itself is a point coordinates [x, y]
                    if len(gen_val) == 2 and all(isinstance(v, (int, float)) for v in gen_val):
                        gen_point = tuple(gen_val)  # Convert [x, y] to (x, y)
                    # Case 2: The list contains points [(x, y), ...]
                    elif isinstance(gen_val[0], tuple):
                        gen_point = gen_val[0]
                    # Case 3: The list contains coordinate pairs as lists [[x, y], ...]
                    elif isinstance(gen_val[0], list) and len(gen_val[0]) == 2:
                        gen_point = tuple(gen_val[0])  # Convert [x, y] to (x, y)
                    else:
                        return False, is_binary, parsed_answer, is_parsable
                elif isinstance(gen_val, tuple):
                    gen_point = gen_val
                else:
                    return False, is_binary, parsed_answer, is_parsable

                if not (isinstance(gen_point, tuple) and len(gen_point) == 2):
                    return False, is_binary, parsed_answer, is_parsable
                
                x, y = float(gen_point[0]), float(gen_point[1])
                parsed_answer = (x, y)
                is_parsable = True
                correct = point_in_polygon(x, y, gt_polygon)
                return correct, is_binary, parsed_answer, is_parsable
            except Exception:
                # If all parsing attempts fail, return False
                return False, is_binary, parsed_answer, is_parsable
                
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return False, is_binary, parsed_answer, is_parsable

def eval_pregenerated_results(gt_data, results_data, data_dir):
    """
    Evaluate pre-generated results against ground truth.
    
    Args:
        gt_data: List of ground truth data (from the benchmark)
        results_data: List of pre-generated model responses
        data_dir: Root directory containing dataset files and images
        
    Returns:
        Dictionary of evaluation statistics
    """
    results = []
    num_correct = 0
    num_total = 0  # Will count only entries that can be evaluated
    illformed_questions = 0
    illformed_responses = 0
    unmatched_entries = 0  # New counter for entries without matching results
    
    # Dictionary to keep per-category statistics
    category_stats = {}

    # Pre-process results_data for more efficient matching
    # Build lookup dictionaries for faster matching
    results_by_idx = {}
    
    for result_entry in results_data:
        idx = result_entry.get("idx")
        if idx is not None:
            results_by_idx[idx] = result_entry

    # Process ground truth entries
    for idx, gt_entry in enumerate(tqdm(gt_data, desc="Evaluating Pre-generated Results")):
        # Extract data from ground truth entry
        question = gt_entry.get("question", "")
        ground_truth = gt_entry.get("answer", "")
        
        if not question or not ground_truth:
            illformed_questions += 1
            continue

        # Increment category stats
        category = gt_entry.get("category", "unknown")
        if category not in category_stats:
            category_stats[category] = {"num_correct": 0, "num_total": 0}
        
        # Try to find a match in the pre-processed results
        matched_result = results_by_idx.get(idx)
        
        # If no match found, count as unmatched
        if matched_result is None:
            unmatched_entries += 1
            continue
        
        # Only now do we count this entry toward the total and category
        num_total += 1
        category_stats[category]["num_total"] += 1
        
        # Extract generated answer
        generated_answer = matched_result.get("prediction", "")
        
        if not generated_answer:
            illformed_responses += 1
            continue

        # Evaluate the answer
        correct, is_binary, parsed_answer, is_parsable = evaluate_answer(ground_truth, generated_answer)
        
        # Count illformed responses - now tracks any answer that couldn't be parsed correctly
        if not is_parsable:
            illformed_responses += 1

        if correct:
            num_correct += 1
            category_stats[category]["num_correct"] += 1

        results.append({
            "idx": idx,
            "question": question,
            "expected_answer": ground_truth,
            "generated_answer": generated_answer,
            "parsed_answer": str(parsed_answer) if parsed_answer is not None else None,
            "correct": correct,
            "is_parsable": is_parsable,
            "category": category
        })

    # Calculate accuracy
    accuracy = 100.0 * num_correct / num_total if num_total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
        "illformed_questions": illformed_questions,
        "illformed_responses": illformed_responses,
        "unmatched_entries": unmatched_entries,
        "category_stats": category_stats,
        "results": results
    }

def main():
    """Main evaluation function"""
    
    # Load the original RoboSpatial-Home dataset
    print("Loading RoboSpatial-Home dataset...")
    ds = load_dataset("chanhee-luke/RoboSpatial-Home")
    
    # Combine all splits into a single list for evaluation
    gt_data = []
    for split_name in ['context', 'compatibility', 'configuration']:
        for item in ds[split_name]:
            gt_data.append(item)
    
    print(f"Loaded {len(gt_data)} ground truth entries")
    
    # Load prediction results
    results_file = "/mnt/chensenda/codes/VLN/InternVL/internvl_chat/eval/robospatial/robospatial_predictions_2.3.json"
    print(f"Loading prediction results from {results_file}...")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    print(f"Loaded {len(results_data)} prediction results")
    
    # Data directory (not used in this evaluation since we're using predictions)
    data_dir = "/mnt/chensenda/codes/VLN/InternVL/internvl_chat/eval/robospatial"
    
    # Evaluate results
    print("Starting evaluation...")
    eval_results = eval_pregenerated_results(gt_data, results_data, data_dir)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {eval_results['accuracy']:.2f}%")
    print(f"Correct: {eval_results['num_correct']}/{eval_results['num_total']}")
    print(f"Illformed Questions: {eval_results['illformed_questions']}")
    print(f"Illformed Responses: {eval_results['illformed_responses']}")
    print(f"Unmatched Entries: {eval_results['unmatched_entries']}")
    
    # Print per-category results
    print("\nPer-Category Results:")
    print("-" * 30)
    for category, stats in eval_results['category_stats'].items():
        if stats['num_total'] > 0:
            accuracy = 100.0 * stats['num_correct'] / stats['num_total']
            print(f"{category}: {accuracy:.2f}% ({stats['num_correct']}/{stats['num_total']})")
    
    # Save detailed results
    output_file = "/mnt/chensenda/codes/VLN/InternVL/internvl_chat/eval/robospatial/eval_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main() 