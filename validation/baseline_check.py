import pandas as pd
import os
import numpy as np
from validation.utils import write_text


def run():
    matrix_path = 'Event_occurrence_matrix.csv'
    summary_path = 'artifacts/validation/occurrence_matrix_summary.md'
    
    if not os.path.exists(matrix_path):
        write_text(summary_path, 'Skipped: Event_occurrence_matrix.csv not found')
        return {'skipped': True, 'summary_path': summary_path}
    
    try:
        df = pd.read_csv(matrix_path)
        
        is_square = len(df) == len(df.columns)
        is_symmetric = False
        
        if is_square:
            try:
                matrix = df.values
                if matrix.shape[0] == matrix.shape[1]:
                    is_symmetric = np.allclose(matrix, matrix.T)
            except:
                pass
        
        diagonal_sum = 0
        if is_square:
            try:
                diagonal_sum = np.trace(df.values)
            except:
                pass
        
        summary_content = f"""# Occurrence Matrix Summary

## Statistics

- Shape: {df.shape[0]} x {len(df.columns)}
- Is square: {'✓' if is_square else '✗'}
- Is symmetric: {'✓' if is_symmetric else '✗'}
- Diagonal sum: {diagonal_sum}

## Validation

- Matrix is square: {'✓' if is_square else '✗'}
- Matrix is symmetric: {'✓' if is_symmetric else '✗'}
"""
        
        write_text(summary_path, summary_content)
        
        return {
            'summary_path': summary_path
        }
    
    except Exception as e:
        write_text(summary_path, f'Error processing matrix: {str(e)}')
        return {'error': str(e), 'summary_path': summary_path}

