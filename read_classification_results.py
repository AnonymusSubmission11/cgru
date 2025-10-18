#!/usr/bin/env python3
"""
Script to read and print classification results saved by classification.py

This script supports UnlearnCanvas metrics for evaluating object unlearning:
- Unlearning Accuracy (UA): Performance on target object class (lower is better for unlearning)
- In-Domain Retain Accuracy (IRA): Performance on other object classes (higher is better)

Focus on object removal/unlearning evaluation.
"""

import argparse
import os
import torch
from typing import Dict, Any
import json

def print_section_header(title: str, width: int = 80):
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

def print_subsection_header(title: str, width: int = 60):
    """Print a formatted subsection header"""
    print(f"\n{'-' * width}")
    print(f"{title:^{width}}")
    print(f"{'-' * width}")

def format_number(value, precision: int = 4):
    """Format numbers for consistent display"""
    if isinstance(value, torch.Tensor):
        value = value.item()
    return f"{value:.{precision}f}"

def print_class_metrics(results: Dict[str, Any], metric_name: str, unit: str = ""):
    """Print metrics for all classes in a formatted table"""
    print_subsection_header(f"{metric_name.upper()} BY CLASS")
    
    print(f"{'Class':<15} {metric_name.capitalize():<15}")
    print("-" * 30)
    
    total = 0
    count = 0
    
    for class_name, value in results[metric_name].items():
        formatted_value = format_number(value)
        print(f"{class_name:<15} {formatted_value:<15}{unit}")
        
        # Calculate average (skip for misclassified)
        if metric_name != "misclassified":
            if isinstance(value, torch.Tensor):
                total += value.item()
            else:
                total += value
            count += 1
    
    if count > 0:
        avg = total / count
        print("-" * 30)
        print(f"{'Average':<15} {format_number(avg):<15}{unit}")

def print_misclassification_matrix(misclassified: Dict[str, Dict[str, int]]):
    """Print detailed misclassification matrix"""
    print_subsection_header("MISCLASSIFICATION MATRIX")
    
    # Get all classes
    classes = list(misclassified.keys())
    
    # Print header
    print(f"{'True Class':<15}", end="")
    for pred_class in classes:
        print(f"{pred_class[:8]:<10}", end="")
    print()
    
    print("-" * (15 + 10 * len(classes)))
    
    # Print each row
    for true_class in classes:
        print(f"{true_class:<15}", end="")
        for pred_class in classes:
            count = misclassified[true_class][pred_class]
            print(f"{count:<10}", end="")
        print()

def print_top_misclassifications(misclassified: Dict[str, Dict[str, int]], top_n: int = 5):
    """Print the most common misclassifications"""
    print_subsection_header(f"TOP {top_n} MISCLASSIFICATIONS")
    
    # Collect all misclassifications (excluding correct predictions)
    misclassifications = []
    for true_class, predictions in misclassified.items():
        for pred_class, count in predictions.items():
            if true_class != pred_class and count > 0:
                misclassifications.append((true_class, pred_class, count))
    
    # Sort by count (descending)
    misclassifications.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'True Class':<15} {'Predicted As':<15} {'Count':<10}")
    print("-" * 40)
    
    for true_class, pred_class, count in misclassifications[:top_n]:
        print(f"{true_class:<15} {pred_class:<15} {count:<10}")

def calculate_unlearning_metrics(results: Dict[str, Any], target_concept: str):
    """Calculate UnlearnCanvas metrics for object unlearning (UA, IRA)"""
    
    # Available object classes (from the original classification script)
    class_available = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers",
                      "Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers",
                      "Trees", "Waterfalls"]
    
    # Object unlearning: target is an object class
    if target_concept not in class_available:
        print(f"Warning: Target object class '{target_concept}' not found in available classes")
        return None
        
    # UA: Accuracy on target object class (we want this LOW for successful unlearning)
    ua = results["acc"][target_concept]
    
    # IRA: Average accuracy on other object classes (retain these)
    other_classes = [cls for cls in class_available if cls != target_concept]
    ira = sum(results["acc"][cls] for cls in other_classes) / len(other_classes)
    
    metrics = {
        "unlearning_accuracy": float(ua),
        "in_domain_retain_accuracy": float(ira),
        "target_concept": target_concept,
        "unlearning_success_rate": float(1 - ua),  # Higher is better for unlearning
        "num_retain_classes": len(other_classes)
    }
    
    return metrics

def print_summary_stats(results: Dict[str, Any], unlearning_metrics: Dict = None):
    """Print overall summary statistics"""
    print_subsection_header("SUMMARY STATISTICS")
    
    # Calculate overall stats
    total_loss = sum(v.item() if isinstance(v, torch.Tensor) else v for v in results["loss"].values())
    total_acc = sum(v.item() if isinstance(v, torch.Tensor) else v for v in results["acc"].values())
    total_pred_loss = sum(v.item() if isinstance(v, torch.Tensor) else v for v in results["pred_loss"].values())
    
    num_classes = len(results["loss"])
    
    avg_loss = total_loss / num_classes
    avg_acc = total_acc / num_classes
    avg_pred_loss = total_pred_loss / num_classes
    
    # Total misclassifications
    total_misclass = sum(sum(predictions.values()) for predictions in results["misclassified"].values())
    
    print(f"Test Theme: {results['test_theme']}")
    print(f"Input Directory: {results['input_dir']}")
    print(f"Number of Classes: {num_classes}")
    print(f"Total Classifications: {total_misclass}")
    print()
    print("KEY METRICS:")
    print(f"  Average Accuracy: {format_number(avg_acc * 100, 2)}% ({format_number(avg_acc)})")
    print(f"  Average Loss: {format_number(avg_loss)}")
    print(f"  Average Prediction Loss: {format_number(avg_pred_loss)}")
    
    # Print unlearning metrics if available
    if unlearning_metrics:
        print_unlearning_metrics(unlearning_metrics)

def print_unlearning_metrics(metrics: Dict):
    """Print UnlearnCanvas object unlearning metrics"""
    print_subsection_header("OBJECT UNLEARNING METRICS (UnlearnCanvas)")
    
    print(f"Target Object Class: {metrics['target_concept']}")
    print(f"Retain Classes Evaluated: {metrics['num_retain_classes']}")
    print()
    
    ua = metrics['unlearning_accuracy']
    print(f"Unlearning Accuracy (UA): {format_number(ua * 100, 2)}% ({format_number(ua)})")
    print(f"  → Unlearning Success Rate: {format_number(metrics['unlearning_success_rate'] * 100, 2)}%")
    print(f"     (Lower UA = Better unlearning)")
    
    ira = metrics['in_domain_retain_accuracy']
    print(f"In-Domain Retain Accuracy (IRA): {format_number(ira * 100, 2)}% ({format_number(ira)})")
    print(f"  → Average accuracy on {metrics['num_retain_classes']} other object classes")
    
    print("\nInterpretation:")
    print("  • UA should be LOW (poor performance on target object class)")
    print("  • IRA should be HIGH (good performance on other object classes)")
    print(f"  • Gap (IRA - UA): {format_number((ira - ua) * 100, 2)}% (larger gap = better unlearning)")

def print_results(results: Dict[str, Any], show_detailed: bool = True, target_concept: str = None):
    """Print all results in a formatted way"""
    
    print_section_header(f"CLASSIFICATION RESULTS - {results['test_theme'].upper()}")
    
    # Calculate unlearning metrics if target concept is specified
    unlearning_metrics = None
    if target_concept:
        unlearning_metrics = calculate_unlearning_metrics(results, target_concept)
    
    # Summary stats
    print_summary_stats(results, unlearning_metrics)
    
    # Individual metrics
    print_class_metrics(results, "acc", "%")
    print_class_metrics(results, "loss")
    print_class_metrics(results, "pred_loss")
    
    if show_detailed:
        # Misclassification analysis
        print_top_misclassifications(results["misclassified"])
        print_misclassification_matrix(results["misclassified"])
    
    return unlearning_metrics

def save_results_as_json(results: Dict[str, Any], output_path: str, unlearning_metrics: Dict = None):
    """Save results as JSON for easier analysis"""
    # Convert tensors to regular numbers for JSON serialization
    json_results = {}
    for key, value in results.items():
        if key == "misclassified":
            json_results[key] = value
        elif isinstance(value, dict):
            json_results[key] = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in value.items()}
        else:
            json_results[key] = value
    
    # Calculate and add average metrics
    num_classes = len(results["loss"])
    
    total_loss = sum(v.item() if isinstance(v, torch.Tensor) else v for v in results["loss"].values())
    total_acc = sum(v.item() if isinstance(v, torch.Tensor) else v for v in results["acc"].values())
    total_pred_loss = sum(v.item() if isinstance(v, torch.Tensor) else v for v in results["pred_loss"].values())
    
    json_results["averages"] = {
        "avg_loss": total_loss / num_classes,
        "avg_accuracy": total_acc / num_classes,
        "avg_pred_loss": total_pred_loss / num_classes,
        "num_classes": num_classes
    }
    
    # Add unlearning metrics if available
    if unlearning_metrics:
        json_results["unlearning_metrics"] = unlearning_metrics
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved as JSON to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Read and print classification results with object unlearning metrics")
    parser.add_argument("--result_file", type=str, required=True, 
                       help="Path to the .pth result file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory containing multiple result files")
    parser.add_argument("--save_json", action="store_true",
                       help="Save results as JSON file")
    parser.add_argument("--detailed", action="store_true", default=True,
                       help="Show detailed misclassification analysis")
    parser.add_argument("--summary_only", action="store_true",
                       help="Show only summary statistics")
    
    # Object unlearning evaluation arguments
    parser.add_argument("--target_concept", type=str, default=None,
                       help="Target object class to evaluate for unlearning")
    parser.add_argument("--list_concepts", action="store_true",
                       help="List all available object classes and exit")
    
    args = parser.parse_args()
    
    # List available object classes if requested
    if args.list_concepts:
        class_available = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers",
                          "Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers",
                          "Trees", "Waterfalls"]
        
        print("AVAILABLE OBJECT CLASSES FOR UNLEARNING EVALUATION")
        print("=" * 50)
        for i, cls in enumerate(class_available, 1):
            print(f"{i:2d}. {cls}")
        
        print("\nExample usage:")
        print("  # Evaluate object unlearning for Cats")
        print("  python read_classification_results.py --result_file result.pth --target_concept Cats")
        print("  # Process all results with Cat unlearning evaluation")
        print("  python read_classification_results.py --output_dir results/ --target_concept Cats --save_json")
        return
    
    if args.output_dir:
        # Process all .pth files in the directory
        pth_files = [f for f in os.listdir(args.output_dir) if f.endswith('.pth')]
        
        if not pth_files:
            print(f"No .pth files found in {args.output_dir}")
            return
        
        for pth_file in sorted(pth_files):
            result_path = os.path.join(args.output_dir, pth_file)
            try:
                results = torch.load(result_path, map_location='cpu')
                unlearning_metrics = print_results(results, 
                                                  show_detailed=not args.summary_only,
                                                  target_concept=args.target_concept)
                
                if args.save_json:
                    json_path = result_path.replace('.pth', '.json')
                    save_results_as_json(results, json_path, unlearning_metrics)
                    
            except Exception as e:
                print(f"Error loading {result_path}: {e}")
    
    else:
        # Process single file
        try:
            results = torch.load(args.result_file, map_location='cpu')
            unlearning_metrics = print_results(results, 
                                              show_detailed=not args.summary_only,
                                              target_concept=args.target_concept)
            
            if args.save_json:
                json_path = args.result_file.replace('.pth', '.json')
                save_results_as_json(results, json_path, unlearning_metrics)
                
        except Exception as e:
            print(f"Error loading {args.result_file}: {e}")

if __name__ == "__main__":
    main()
