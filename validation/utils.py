import os
import random
import json
import yaml
import csv
import re
import chardet
from pathlib import Path
from typing import Dict, List, Any


def set_seed(seed: int):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)


def ensure_dirs(*paths):
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_text_lines(path, sample_head=20, sample_rand=20, seed=1337):
    random.seed(seed)
    lines = []
    head_lines = []
    rand_lines = []
    
    try:
        with open(path, 'rb') as f:
            raw = f.read()
            result = chardet.detect(raw)
            encoding = result.get('encoding', 'utf-8')
        
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            all_lines = f.readlines()
            total = len(all_lines)
            
            head_lines = all_lines[:sample_head]
            
            if total > sample_head:
                rand_indices = random.sample(range(sample_head, total), min(sample_rand, total - sample_head))
                rand_lines = [all_lines[i] for i in sorted(rand_indices)]
    
    except Exception as e:
        pass
    
    return {
        'total_lines': len(all_lines) if 'all_lines' in locals() else 0,
        'head': head_lines,
        'random': rand_lines,
        'encoding': encoding if 'encoding' in locals() else 'utf-8'
    }


def detect_encoding_and_newlines(path) -> dict:
    try:
        with open(path, 'rb') as f:
            raw = f.read(10000)
            result = chardet.detect(raw)
            encoding = result.get('encoding', 'utf-8')
        
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            sample = f.read(10000)
            if '\r\n' in sample:
                newline = '\r\n'
            elif '\n' in sample:
                newline = '\n'
            else:
                newline = ''
        
        return {'encoding': encoding, 'newline': newline}
    except Exception as e:
        return {'encoding': 'utf-8', 'newline': '\n'}


def write_json(path, obj):
    ensure_dirs(path)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def write_yaml(path, obj):
    ensure_dirs(path)
    with open(path, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)


def write_csv_rows(path, rows: List[Dict], fieldnames: List[str]):
    ensure_dirs(path)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, s: str):
    ensure_dirs(path)
    with open(path, 'w') as f:
        f.write(s)


def save_histogram(path_png, data: List[int], title: str, xlabel: str, ylabel: str):
    import matplotlib.pyplot as plt
    import numpy as np
    
    ensure_dirs(path_png)
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150, bbox_inches='tight')
    plt.close()


def save_bar_topn(path_png, labels: List[str], values: List[int], title: str, top: int = 30):
    import matplotlib.pyplot as plt
    
    ensure_dirs(path_png)
    
    if len(labels) > top:
        pairs = list(zip(labels, values))
        pairs.sort(key=lambda x: x[1], reverse=True)
        labels, values = zip(*pairs[:top])
        labels, values = list(labels), list(values)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(labels)), values, edgecolor='black', alpha=0.7)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.title(title)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(path_png, dpi=150, bbox_inches='tight')
    plt.close()

