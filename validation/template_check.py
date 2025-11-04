import pandas as pd
import os
from validation.utils import write_text, write_csv_rows


def run():
    template_path = 'HDFS.log_templates.csv'
    traces_path = 'Event_traces.csv'
    summary_path = 'artifacts/validation/template_summary.md'
    unseen_path = 'artifacts/validation/unseen_templates.tsv'
    
    if not os.path.exists(template_path):
        return {'error': 'Template file not found'}
    
    df_templates = pd.read_csv(template_path)
    
    id_col = None
    template_col = None
    
    for col in df_templates.columns:
        if col.lower() in ['eventid', 'event_id', 'id', 'log_key']:
            id_col = col
        if col.lower() in ['eventtemplate', 'event_template', 'template']:
            template_col = col
    
    if id_col is None or template_col is None:
        return {'error': 'Could not find template columns'}
    
    unique_templates = len(df_templates)
    templates_with_placeholders = 0
    duplicate_texts = []
    
    template_texts = {}
    for idx, row in df_templates.iterrows():
        template_text = str(row[template_col])
        if '*' in template_text or '<*>' in template_text:
            templates_with_placeholders += 1
        if template_text in template_texts:
            duplicate_texts.append({
                'id1': template_texts[template_text],
                'id2': row[id_col],
                'template': template_text[:100]
            })
        else:
            template_texts[template_text] = row[id_col]
    
    if os.path.exists(traces_path):
        df_traces = pd.read_csv(traces_path)
        features_col = None
        for col in df_traces.columns:
            if col.lower() in ['features', 'sequence']:
                features_col = col
                break
        
        if features_col:
            all_event_ids = set()
            for seq in df_traces[features_col]:
                try:
                    if isinstance(seq, str):
                        if seq.startswith('['):
                            import json
                            seq_list = json.loads(seq)
                        else:
                            seq_list = [x.strip() for x in seq.split(',')]
                        all_event_ids.update(seq_list)
                except:
                    pass
            
            template_ids_in_map = set(df_templates[id_col].astype(str))
            unused_templates = []
            missing_ids = []
            
            for tid in template_ids_in_map:
                if tid not in all_event_ids:
                    template_row = df_templates[df_templates[id_col].astype(str) == tid]
                    if not template_row.empty:
                        unused_templates.append({
                            'template_id': tid,
                            'template': template_row.iloc[0][template_col][:100]
                        })
            
            for eid in all_event_ids:
                if eid not in template_ids_in_map:
                    missing_ids.append({'event_id': eid})
            
            unseen_rows = unused_templates + [{'template_id': 'MISSING', 'template': f'Event {m["event_id"]} not in templates'} for m in missing_ids]
            write_csv_rows(unseen_path, unseen_rows, ['template_id', 'template'])
        else:
            write_csv_rows(unseen_path, [], ['template_id', 'template'])
    else:
        write_csv_rows(unseen_path, [], ['template_id', 'template'])
    
    summary_content = f"""# Template Validation Summary

## Statistics

- Unique templates: {unique_templates}
- Templates with placeholders (* or <*>): {templates_with_placeholders}
- Duplicate template texts: {len(duplicate_texts)}

## Validation

- Template count in range [25, 35]: {'✓' if 25 <= unique_templates <= 35 else '✗'} ({unique_templates})
- All templates have placeholders: {'✓' if templates_with_placeholders == unique_templates else '✗'} ({templates_with_placeholders}/{unique_templates})
- No duplicate texts: {'✓' if len(duplicate_texts) == 0 else '✗'} ({len(duplicate_texts)} duplicates)
"""
    
    write_text(summary_path, summary_content)
    
    return {
        'summary_path': summary_path,
        'unseen_path': unseen_path
    }

