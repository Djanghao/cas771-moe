#!/usr/bin/env python3
import os
import subprocess
import time
import platform

def open_file(filepath):
    """
    Open a file with the default application in a cross-platform way.
    """
    system = platform.system()
    try:
        if system == 'Darwin':  # macOS
            subprocess.Popen(['open', filepath])
        elif system == 'Windows':  # Windows
            os.startfile(filepath)
        elif system == 'Linux':  # Linux
            subprocess.Popen(['xdg-open', filepath])
        else:
            print(f"Unsupported platform: {system}. Cannot open file automatically.")
    except Exception as e:
        print(f"Error opening file {filepath}: {e}")

def run_evaluation():
    """
    Run both Task A and Task B evaluation scripts sequentially
    and open the generated images.
    """
    print("=" * 50)
    print("Starting evaluation for Task A...")
    print("=" * 50)
    
    # Run Task A evaluation
    subprocess.run(["python", "evaluate_task_a.py"], check=True)
    
    # Wait a moment to ensure files are saved
    time.sleep(1)
    
    # Open Task A images
    task_a_sample_predictions = "evaluation/TaskA/sample_predictions.png"
    task_a_expert_analysis = "evaluation/TaskA/expert_analysis.png"
    
    if os.path.exists(task_a_sample_predictions):
        print(f"\nOpening {task_a_sample_predictions}...")
        open_file(task_a_sample_predictions)
    
    if os.path.exists(task_a_expert_analysis):
        print(f"Opening {task_a_expert_analysis}...")
        open_file(task_a_expert_analysis)
    
    # Wait a moment before starting Task B
    time.sleep(2)
    
    print("\n" + "=" * 50)
    print("Starting evaluation for Task B...")
    print("=" * 50)
    
    # Run Task B evaluation
    subprocess.run(["python", "evaluate_task_b.py"], check=True)
    
    # Wait a moment to ensure files are saved
    time.sleep(1)
    
    # Open Task B images
    task_b_sample_predictions = "evaluation/TaskB/sample_predictions.png"
    task_b_expert_analysis = "evaluation/TaskB/expert_analysis.png"
    
    if os.path.exists(task_b_sample_predictions):
        print(f"\nOpening {task_b_sample_predictions}...")
        open_file(task_b_sample_predictions)
    
    if os.path.exists(task_b_expert_analysis):
        print(f"Opening {task_b_expert_analysis}...")
        open_file(task_b_expert_analysis)
    
    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print("=" * 50)

if __name__ == "__main__":
    run_evaluation()
