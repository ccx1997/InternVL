#!/usr/bin/env python3
"""
Script to evaluate EM-1 and EM-R1 metrics for SQA predictions.
EM-1: Exact Match (normalized with comprehensive answer cleaning)
EM-R1: Relaxed Exact Match (multiple strategies: exact, substring, no-space, word overlap)

Based on original ScanQA evaluation script with comprehensive answer normalization including:
- Common typo corrections (letf->left, tehre->there, etc.)
- Digit to word conversion (1->one, 2->two, etc.)
- Article removal (a, an, the)
- Punctuation and whitespace normalization

Usage:
    python eval_sqa3d.py --pred "path/to/predictions.json"
"""

import json
import re
import string
import argparse
import os
from typing import List, Dict, Tuple


def normalize_answer(text: str) -> str:
    """
    Comprehensive answer normalization based on original ScanQA evaluation script.
    Handles typos, digit-to-word conversion, article removal, and other cleanups.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove trailing and leading whitespace
    text = re.sub('[ ]+$', '', text)
    text = re.sub('^[ ]+', '', text)
    
    # Remove multiple spaces
    text = re.sub(' {2,}', ' ', text)
    
    # Fix spacing after periods
    text = re.sub('\.[ ]{2,}', '. ', text)
    
    # Remove unwanted characters (keep letters, numbers, comma, apostrophe, space, dash, colon)
    text = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', text)
    
    # Fix character replacements
    text = re.sub('ç', 'c', text)
    text = re.sub('\'', '\'', text)
    
    # Fix common typos
    text = re.sub(r'\bletf\b', 'left', text)
    text = re.sub(r'\blet\b', 'left', text)
    text = re.sub(r'\btehre\b', 'there', text)
    text = re.sub(r'\brigth\b', 'right', text)
    text = re.sub(r'\brght\b', 'right', text)
    text = re.sub(r'\bbehine\b', 'behind', text)
    text = re.sub(r'\btv\b', 'TV', text)
    text = re.sub(r'\bchai\b', 'chair', text)
    text = re.sub(r'\bwasing\b', 'washing', text)
    text = re.sub(r'\bwaslked\b', 'walked', text)
    text = re.sub(r'\boclock\b', 'o\'clock', text)
    text = re.sub(r'\bo\'[ ]+clock\b', 'o\'clock', text)
    
    # Convert digits to words
    text = re.sub(r'\b0\b', 'zero', text)
    text = re.sub(r'\bnone\b', 'zero', text)
    text = re.sub(r'\b1\b', 'one', text)
    text = re.sub(r'\b2\b', 'two', text)
    text = re.sub(r'\b3\b', 'three', text)
    text = re.sub(r'\b4\b', 'four', text)
    text = re.sub(r'\b5\b', 'five', text)
    text = re.sub(r'\b6\b', 'six', text)
    text = re.sub(r'\b7\b', 'seven', text)
    text = re.sub(r'\b8\b', 'eight', text)
    text = re.sub(r'\b9\b', 'nine', text)
    text = re.sub(r'\b10\b', 'ten', text)
    text = re.sub(r'\b11\b', 'eleven', text)
    text = re.sub(r'\b12\b', 'twelve', text)
    text = re.sub(r'\b13\b', 'thirteen', text)
    text = re.sub(r'\b14\b', 'fourteen', text)
    text = re.sub(r'\b15\b', 'fifteen', text)
    text = re.sub(r'\b16\b', 'sixteen', text)
    text = re.sub(r'\b17\b', 'seventeen', text)
    text = re.sub(r'\b18\b', 'eighteen', text)
    text = re.sub(r'\b19\b', 'nineteen', text)
    text = re.sub(r'\b20\b', 'twenty', text)
    text = re.sub(r'\b23\b', 'twenty-three', text)
    
    # Remove numbers after letters (no1 -> no, mat2 -> mat)
    text = re.sub(r'\b([a-zA-Z]+)([0-9])\b', r'\g<1>', text)
    
    # Remove articles (a, an, the)
    text = re.sub(r'\ba\b ([a-zA-Z]+)', r'\g<1>', text)
    text = re.sub(r'\ban\b ([a-zA-Z]+)', r'\g<1>', text)
    text = re.sub(r'\bthe\b ([a-zA-Z]+)', r'\g<1>', text)
    
    # Fix additional word forms
    text = re.sub(r'\bbackwards\b', 'backward', text)
    
    # Final cleanup of whitespace
    text = ' '.join(text.split()).strip()
    
    return text


def exact_match_strict(pred: str, gt: str) -> bool:
    """
    Exact Match (EM-1): Case-sensitive exact matching after normalization
    """
    return normalize_answer(pred) == normalize_answer(gt)


def exact_match_relaxed(pred: str, gt: str) -> bool:
    """
    Exact Match Relaxed (EM-R1): More flexible matching strategies
    Based on original ScanQA evaluation script
    """
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    
    # Strategy 1: Exact match after normalization
    if pred_norm == gt_norm:
        return True
    
    # Strategy 2: Prediction is contained in ground truth
    if pred_norm in gt_norm:
        return True
    
    # Strategy 3: Prediction without spaces is contained in ground truth without spaces
    pred_no_space = ''.join(pred_norm.split())
    gt_no_space = ''.join(gt_norm.split())
    if pred_no_space in gt_no_space:
        return True
    
    # Strategy 4: Word overlap - check if they have common words
    pred_words = set(pred_norm.split())
    gt_words = set(gt_norm.split())
    if len(pred_words.intersection(gt_words)) > 0:
        return True
    
    return False


def evaluate_predictions(predictions_file: str) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Evaluate predictions and calculate EM-1 and EM-R1 metrics
    
    Args:
        predictions_file: Path to JSON file containing predictions
        
    Returns:
        Tuple of metrics dictionary and detailed results list
    """
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    em1_correct = 0  # Exact Match normalized
    emr1_correct = 0  # Exact Match relaxed (multiple strategies)
    total_samples = 0
    
    # Statistics for detailed analysis
    results = []
    
    # Debug counters for different matching strategies
    strategy_stats = {
        'exact_match': 0,
        'substring_match': 0,
        'no_space_match': 0,
        'word_overlap': 0
    }
    
    debug_cases = []  # Store some interesting cases for analysis
    
    for item in data:
        conversations = item['conversations']
        
        # Extract ground truth and prediction
        gt_answer = None
        pred_answer = None
        
        for conv in conversations:
            if conv['from'] == 'gpt':
                gt_answer = conv['value']
            elif conv['from'] == 'internVL':
                pred_answer = conv['value']
        
        # Skip if either answer is missing
        if gt_answer is None or pred_answer is None:
            print(f"Warning: Missing answer in item {item['id']}")
            continue
        
        total_samples += 1
        
        # Calculate metrics
        em1_match = exact_match_strict(pred_answer, gt_answer)
        emr1_match = exact_match_relaxed(pred_answer, gt_answer)
        
        if em1_match:
            em1_correct += 1
        if emr1_match:
            emr1_correct += 1
        
        # Determine which strategy worked for relaxed matching
        matched_strategy = None
        if emr1_match:
            pred_norm = normalize_answer(pred_answer)
            gt_norm = normalize_answer(gt_answer)
            
            if pred_norm == gt_norm:
                matched_strategy = 'exact_match'
                strategy_stats['exact_match'] += 1
            elif pred_norm in gt_norm:
                matched_strategy = 'substring_match'
                strategy_stats['substring_match'] += 1
            elif ''.join(pred_norm.split()) in ''.join(gt_norm.split()):
                matched_strategy = 'no_space_match'
                strategy_stats['no_space_match'] += 1
            elif len(set(pred_norm.split()).intersection(set(gt_norm.split()))) > 0:
                matched_strategy = 'word_overlap'
                strategy_stats['word_overlap'] += 1
        
        # Debug: Check cases where EM-1 fails but EM-R1 passes
        if not em1_match and emr1_match:
            if len(debug_cases) < 10:  # Store first 10 cases for analysis
                debug_cases.append({
                    'id': item['id'],
                    'gt': gt_answer,
                    'pred': pred_answer,
                    'gt_norm': normalize_answer(gt_answer),
                    'pred_norm': normalize_answer(pred_answer),
                    'strategy': matched_strategy
                })
        
        # Store result for detailed analysis
        results.append({
            'id': item['id'],
            'question': next((c['value'] for c in conversations if c['from'] == 'human'), ''),
            'ground_truth': gt_answer,
            'prediction': pred_answer,
            'em1_match': em1_match,
            'emr1_match': emr1_match,
            'matched_strategy': matched_strategy,
            'gt_normalized': normalize_answer(gt_answer),
            'pred_normalized': normalize_answer(pred_answer),
            'gt_stripped': gt_answer.strip(),
            'pred_stripped': pred_answer.strip()
        })
    
    # Calculate final metrics
    em1_score = em1_correct / total_samples if total_samples > 0 else 0.0
    emr1_score = emr1_correct / total_samples if total_samples > 0 else 0.0
    
    metrics = {
        'total_samples': total_samples,
        'em1_correct': em1_correct,
        'emr1_correct': emr1_correct,
        'em1_score': em1_score,
        'emr1_score': emr1_score,
        'strategy_stats': strategy_stats,
        'debug_cases': debug_cases
    }
    
    return metrics, results


def analyze_differences(results: List[Dict]):
    """
    Analyze differences between ground truth and predictions
    """
    print(f"\n=== DETAILED ANALYSIS ===")
    
    # Find cases with different formatting
    case_differences = []
    punct_differences = []
    space_differences = []
    
    for r in results:
        gt = r['ground_truth']
        pred = r['prediction']
        
        # Check for case differences
        if gt.lower() == pred.lower() and gt != pred:
            case_differences.append(r)
        
        # Check for punctuation differences
        gt_no_punct = gt.translate(str.maketrans('', '', string.punctuation))
        pred_no_punct = pred.translate(str.maketrans('', '', string.punctuation))
        if gt_no_punct == pred_no_punct and gt != pred:
            punct_differences.append(r)
        
        # Check for whitespace differences
        if gt.strip() == pred.strip() and gt != pred:
            space_differences.append(r)
    
    print(f"Cases with case differences: {len(case_differences)}")
    print(f"Cases with punctuation differences: {len(punct_differences)}")
    print(f"Cases with whitespace differences: {len(space_differences)}")
    
    # Show examples
    if case_differences:
        print(f"\nCase difference examples:")
        for i, case in enumerate(case_differences[:3]):
            print(f"  {i+1}. GT: '{case['ground_truth']}' | Pred: '{case['prediction']}'")
    
    if punct_differences:
        print(f"\nPunctuation difference examples:")
        for i, case in enumerate(punct_differences[:3]):
            print(f"  {i+1}. GT: '{case['ground_truth']}' | Pred: '{case['prediction']}'")
    
    if space_differences:
        print(f"\nWhitespace difference examples:")
        for i, case in enumerate(space_differences[:3]):
            print(f"  {i+1}. GT: '{case['ground_truth']}' | Pred: '{case['prediction']}'")


def print_sample_errors(results: List[Dict], num_samples: int = 5):
    """
    Print sample errors for analysis
    """
    print(f"\n=== Sample Errors (showing first {num_samples}) ===")
    
    em1_errors = [r for r in results if not r['em1_match']]
    emr1_errors = [r for r in results if not r['emr1_match']]
    
    print(f"\nEM-1 Errors (showing first {min(num_samples, len(em1_errors))}):")
    for i, error in enumerate(em1_errors[:num_samples]):
        print(f"\nError {i+1} (ID: {error['id']}):")
        print(f"  Question: {error['question'][:100]}...")
        print(f"  Ground Truth: '{error['ground_truth']}'")
        print(f"  Prediction:   '{error['prediction']}'")
        print(f"  GT Normalized: '{error['gt_normalized']}'")
        print(f"  Pred Normalized: '{error['pred_normalized']}'")


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Evaluate EM-1 and EM-R1 metrics for SQA3D predictions with comprehensive answer normalization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_sqa3d.py --pred sqa_predictions.json
  python eval_sqa3d.py --pred /path/to/model_outputs.json --output results.json

Evaluation Metrics:
  EM-1:  Exact match after comprehensive normalization (typo fixes, digit->word, article removal)
  EM-R1: Relaxed matching with multiple strategies:
         - Exact match after normalization
         - Substring match (prediction contained in ground truth)
         - No-space match (same as substring but ignoring spaces)
         - Word overlap (at least one common word)
        """
    )
    
    parser.add_argument(
        '--pred', 
        type=str, 
        required=True,
        help='Path to the JSON file containing model predictions'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Path to save detailed evaluation results (default: auto-generated based on input filename)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Show detailed analysis and sample errors'
    )
    
    parser.add_argument(
        '--samples', 
        type=int, 
        default=5,
        help='Number of sample errors to show (default: 5)'
    )
    
    return parser.parse_args()


def main():
    """
    Main evaluation function
    """
    args = parse_args()
    
    predictions_file = args.pred
    
    # Check if file exists
    if not os.path.exists(predictions_file):
        print(f"Error: File '{predictions_file}' not found!")
        return
    
    print("Evaluating SQA3D predictions...")
    print(f"Loading data from: {predictions_file}")
    
    try:
        metrics, results = evaluate_predictions(predictions_file)
        
        # Print results
        print("\n" + "="*60)
        print("SQA3D EVALUATION RESULTS (Comprehensive Normalization)")
        print("="*60)
        print(f"Prediction file: {predictions_file}")
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"\nEM-1 (Exact Match with Normalization):")
        print(f"  Correct: {metrics['em1_correct']}")
        print(f"  Score: {metrics['em1_score']:.4f} ({metrics['em1_score']*100:.2f}%)")
        print(f"\nEM-R1 (Relaxed Match - Multiple Strategies):")
        print(f"  Correct: {metrics['emr1_correct']}")
        print(f"  Score: {metrics['emr1_score']:.4f} ({metrics['emr1_score']*100:.2f}%)")
        
        # Print improvement from relaxed matching
        improvement = metrics['emr1_correct'] - metrics['em1_correct']
        print(f"\nImprovement from relaxed matching: {improvement} samples")
        
        # Show strategy breakdown
        print(f"\nEM-R1 Strategy Breakdown:")
        strategy_stats = metrics['strategy_stats']
        print(f"  Exact match after normalization: {strategy_stats['exact_match']}")
        print(f"  Substring match: {strategy_stats['substring_match']}")
        print(f"  No-space match: {strategy_stats['no_space_match']}")
        print(f"  Word overlap: {strategy_stats['word_overlap']}")
        
        # Show debug cases if any
        if metrics['debug_cases'] and args.verbose:
            print(f"\nCases where relaxed matching helped:")
            for i, case in enumerate(metrics['debug_cases']):
                print(f"  {i+1}. ID {case['id']} (Strategy: {case['strategy']}):")
                print(f"     GT: '{case['gt']}'")
                print(f"     Pred: '{case['pred']}'")
                print(f"     GT normalized: '{case['gt_norm']}'")
                print(f"     Pred normalized: '{case['pred_norm']}'")
                print()
        
        # Show detailed analysis if verbose
        if args.verbose:
            analyze_differences(results)
            print_sample_errors(results, args.samples)
        
        # Save detailed results
        if args.output:
            output_file = args.output
        else:
            # Auto-generate output filename
            base_name = os.path.splitext(os.path.basename(predictions_file))[0]
            output_file = f"{base_name}_evaluation_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'input_file': predictions_file,
                'metrics': metrics,
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")
        
        # Print summary
        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"EM-1:  {metrics['em1_score']*100:.2f}% ({metrics['em1_correct']}/{metrics['total_samples']})")
        print(f"EM-R1: {metrics['emr1_score']*100:.2f}% ({metrics['emr1_correct']}/{metrics['total_samples']})")
        
    except FileNotFoundError:
        print(f"Error: File '{predictions_file}' not found!")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{predictions_file}': {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 