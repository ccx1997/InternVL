#!/usr/bin/env python3
"""
Evaluation script for ScanQA dataset with comprehensive answer normalization
Computes CiDEr, BLEU-1, BLEU-4, METEOR, and ROUGE-L metrics
comparing ground truth answers (from: gpt) with predictions (from: internVL)

Enhanced with comprehensive answer preprocessing including:
- Common typo corrections (letf->left, tehre->there, etc.)
- Digit to word conversion (1->one, 2->two, etc.)
- Article removal (a, an, the)
- Punctuation and whitespace normalization
"""

import json
import argparse
import sys
from typing import List, Dict, Tuple
import numpy as np
import re

# Import required libraries for metrics
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    meteor_available = True
except ImportError:
    print("Please install NLTK: pip install nltk")
    meteor_available = False
    sys.exit(1)

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Please install rouge-score: pip install rouge-score")
    sys.exit(1)

try:
    from pycocoevalcap.cider.cider import Cider
except ImportError:
    print("Warning: CiDEr metric not available. Please install pycocoevalcap: pip install pycocoevalcap")
    cider_available = False
else:
    cider_available = True


class SQA3DEvaluator:
    """
    Enhanced evaluator for ScanQA dataset with comprehensive answer normalization.
    
    Computes standard text generation metrics (BLEU, METEOR, ROUGE-L, CiDEr) with
    improved preprocessing that handles common issues in VQA datasets including:
    - Typo corrections and spelling normalizations
    - Digit-to-word conversions for consistent evaluation
    - Article removal to focus on content words
    - Punctuation and whitespace standardization
    """
    
    def __init__(self):
        self.smoothing_function = SmoothingFunction().method1
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
    def preprocess_text(self, text: str) -> str:
        """
        Comprehensive text preprocessing with normalization based on ScanQA evaluation standards.
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
        text = re.sub(r'\btv\b', 'tv', text)  # Keep lowercase for generation metrics
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
        
        # Remove articles (a, an, the) - but be more careful with generation metrics
        # Only remove articles before nouns (when followed by another word)
        text = re.sub(r'\ba\b ([a-zA-Z]+)', r'\g<1>', text)
        text = re.sub(r'\ban\b ([a-zA-Z]+)', r'\g<1>', text)
        text = re.sub(r'\bthe\b ([a-zA-Z]+)', r'\g<1>', text)
        
        # Fix additional word forms
        text = re.sub(r'\bbackwards\b', 'backward', text)
        
        # Final cleanup of whitespace
        text = ' '.join(text.split()).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using NLTK"""
        return word_tokenize(self.preprocess_text(text))
    
    def compute_bleu_scores(self, references: List[List[str]], hypothesis: List[str]) -> Dict[str, float]:
        """Compute BLEU-1 and BLEU-4 scores"""
        bleu1_scores = []
        bleu4_scores = []
        
        for ref_list, hyp in zip(references, hypothesis):
            # Convert reference list to list of tokenized sentences
            ref_tokens = [self.tokenize_text(ref) for ref in ref_list]
            hyp_tokens = self.tokenize_text(hyp)
            
            # BLEU-1
            bleu1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), 
                                smoothing_function=self.smoothing_function)
            bleu1_scores.append(bleu1)
            
            # BLEU-4
            bleu4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=self.smoothing_function)
            bleu4_scores.append(bleu4)
        
        return {
            'BLEU-1': np.mean(bleu1_scores) * 100,
            'BLEU-4': np.mean(bleu4_scores) * 100
        }
    
    def compute_meteor_scores(self, references: List[List[str]], hypothesis: List[str]) -> float:
        """Compute METEOR scores"""
        if not meteor_available:
            print("Warning: METEOR not available, returning 0.0")
            return 0.0
            
        meteor_scores = []
        total_errors = 0
        
        for i, (ref_list, hyp) in enumerate(zip(references, hypothesis)):
            # METEOR works with tokenized input
            # If multiple references, we compute score against each and take the maximum
            scores = []
            ref_tokens_list = []
            for ref in ref_list:
                # Ensure we have clean text input
                ref_clean = self.preprocess_text(ref)
                # METEOR expects both reference and hypothesis to be tokenized
                ref_tokens = self.tokenize_text(ref_clean)
                ref_tokens_list.append(ref_tokens)
            hyp_clean = self.preprocess_text(hyp)
            # Skip empty strings
            if not ref_clean or not hyp_clean:
                scores.append(0.0)
                continue
            hyp_tokens = self.tokenize_text(hyp_clean)
            score = meteor_score(ref_tokens_list, hyp_tokens)
            # print(f"score: {score}")
            scores.append(score)
            
            meteor_scores.append(max(scores) if scores else 0.0)
        
        successful_scores = [s for s in meteor_scores if s > 0]
        avg_score = np.mean(meteor_scores) * 100
        
        print(f"METEOR computation: {len(successful_scores)}/{len(meteor_scores)} successful (errors: {total_errors})")
        if len(successful_scores) == 0:
            print("Warning: All METEOR computations failed. This might be a NLTK installation issue.")
            print("Try running: python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\"")
        
        return avg_score
    
    def compute_rouge_scores(self, references: List[List[str]], hypothesis: List[str]) -> float:
        """Compute ROUGE-L scores"""
        rouge_scores = []
        
        for ref_list, hyp in zip(references, hypothesis):
            # Compute ROUGE score against each reference and take the maximum
            scores = []
            for ref in ref_list:
                score = self.rouge_scorer.score(self.preprocess_text(ref), self.preprocess_text(hyp))
                scores.append(score['rougeL'].fmeasure)
            
            rouge_scores.append(max(scores) if scores else 0.0)
        
        return np.mean(rouge_scores) * 100
    
    def compute_cider_scores(self, references: List[List[str]], hypothesis: List[str]) -> float:
        """Compute CiDEr scores"""
        if not cider_available:
            print("Warning: CiDEr metric not available")
            return 0.0
        
        # Format data for CiDEr evaluation
        gts = {}  # ground truth
        res = {}  # results
        
        for i, (ref_list, hyp) in enumerate(zip(references, hypothesis)):
            gts[i] = [self.preprocess_text(ref) for ref in ref_list]
            res[i] = [self.preprocess_text(hyp)]
        
        # Debug: Print some sample data to check format
        print(f"CiDEr evaluation: Processing {len(gts)} samples")
        if len(gts) > 0:
            print(f"Sample ground truth: {gts[0]}")
            print(f"Sample prediction: {res[0]}")
        
        try:
            cider_scorer = Cider()
            score, scores = cider_scorer.compute_score(gts, res)
            print(f"CiDEr score computed successfully: {score}")
            return score * 100
        except Exception as e:
            print(f"Warning: CiDEr computation failed with error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            return 0.0
    
    def load_predictions(self, file_path: str) -> Tuple[List[List[str]], List[str]]:
        """Load predictions from JSON file and extract references and hypotheses"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        references = []
        hypotheses = []
        
        for item in data:
            conversations = item['conversations']
            
            # Find ground truth (from: gpt) and prediction (from: internVL)
            ground_truth = None
            prediction = None
            
            for conv in conversations:
                if conv['from'] == 'gpt':
                    ground_truth = conv['value']
                elif conv['from'] == 'internVL':
                    prediction = conv['value']
            
            if ground_truth is not None and prediction is not None:
                # Ensure ground_truth is a list
                if isinstance(ground_truth, str):
                    ground_truth = [ground_truth]
                
                references.append(ground_truth)
                hypotheses.append(prediction)
        
        return references, hypotheses
    
    def evaluate(self, file_path: str) -> Dict[str, float]:
        """Evaluate predictions and return all metrics"""
        print(f"Loading predictions from: {file_path}")
        references, hypotheses = self.load_predictions(file_path)
        
        print(f"Found {len(references)} question-answer pairs")
        
        # Show some examples of preprocessing
        if len(hypotheses) > 0:
            print("\nPreprocessing examples:")
            for i in range(min(3, len(hypotheses))):
                orig_hyp = hypotheses[i]
                orig_ref = references[i][0] if references[i] else ""
                proc_hyp = self.preprocess_text(orig_hyp)
                proc_ref = self.preprocess_text(orig_ref)
                
                if orig_hyp != proc_hyp or orig_ref != proc_ref:
                    print(f"  Example {i+1}:")
                    if orig_ref != proc_ref:
                        print(f"    GT: '{orig_ref}' -> '{proc_ref}'")
                    if orig_hyp != proc_hyp:
                        print(f"    Pred: '{orig_hyp}' -> '{proc_hyp}'")
        
        # Compute all metrics
        results = {}
        
        print("\nComputing BLEU scores...")
        bleu_scores = self.compute_bleu_scores(references, hypotheses)
        results.update(bleu_scores)
        
        print("Computing METEOR scores...")
        meteor_score = self.compute_meteor_scores(references, hypotheses)
        results['METEOR'] = meteor_score
        
        print("Computing ROUGE-L scores...")
        rouge_score = self.compute_rouge_scores(references, hypotheses)
        results['ROUGE-L'] = rouge_score
        
        if cider_available:
            print("Computing CiDEr scores...")
            cider_score = self.compute_cider_scores(references, hypotheses)
            results['CiDEr'] = cider_score
        else:
            results['CiDEr'] = 0.0
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ScanQA predictions with enhanced answer normalization (BLEU, METEOR, ROUGE-L, CiDEr)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced preprocessing includes:
- Common typo corrections and digit-to-word conversion
- Article removal and punctuation normalization
- Consistent whitespace handling

This provides more robust evaluation by normalizing both ground truth and predictions
before computing generation metrics.
        """
    )
    parser.add_argument('--pred', '-p', type=str, required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save evaluation results (optional)')
    
    args = parser.parse_args()
    
    evaluator = SQA3DEvaluator()
    results = evaluator.evaluate(args.pred)
    
    # Print results
    print("\n" + "="*50)
    print("ScanQA EVALUATION RESULTS (Enhanced Normalization)")
    print("="*50)
    for metric, score in results.items():
        print(f"{metric:12}: {score:.2f}")
    print("="*50)
    
    # Save results if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main() 


#     #!/bin/bash

# # Install dependencies if needed
# echo "Installing required packages..."
# pip install -r requirements_eval.txt

# # Run evaluation
# echo "Running SQA3D evaluation..."
# python evaluate_scanqa.py --predictions scanqa_predictions.json --output evaluation_results.json

# echo "Evaluation completed!"
# echo "Results saved to evaluation_results.json" 
