import torch
import os
import argparse
import numpy as np

def check_labels(data_dir):
    print(f"Checking labels in {data_dir}...")
    
    paths = {
        'goals_lateral.pt': ['lane_keep', 'lane_change_left', 'lane_change_right', 'turn_left', 'turn_right', 'stop', 'overtake'],
        'goals_longitudinal.pt': ['accelerate', 'decelerate', 'cruise', 'stop']
    }
    
    for filename, classes in paths.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"[WARN] {filename} not found.")
            continue
            
        labels = torch.load(path)
        print(f"\n--- {filename} ({len(labels)} samples) ---")
        counts = np.bincount(labels.numpy(), minlength=len(classes))
        
        total = len(labels)
        for i, count in enumerate(counts):
            if i < len(classes):
                name = classes[i]
                pct = count / total * 100
                print(f"  {i}: {name:<20} {count:>6} ({pct:.2f}%)")
            else:
                print(f"  {i}: UNKNOWN             {count:>6}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='planner1/data/processed')
    args = parser.parse_args()
    check_labels(args.data_dir)
