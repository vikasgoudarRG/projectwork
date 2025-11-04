"""
Workflow visualization: Event transition graph.
Build directed graph from event sequences.
Highlight anomalous transitions (edges that appear only in anomalous sessions).
"""
import os
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

from src.data.dataset_loader import load_event_traces, split_by_blockid


def visualize_workflow():
    """Generate workflow graph visualization."""
    # Create output directories
    os.makedirs('artifacts/visual', exist_ok=True)
    
    print("\n[1/4] Loading data...")
    
    # Load event traces
    df = load_event_traces('Event_traces.csv')
    
    # Load detection results
    session_df = pd.read_csv('artifacts/detection/session_anomalies.csv')
    anomalous_blocks = set(session_df[session_df['fused_anomaly'] == 1]['block_id'].values)
    
    print(f"  Total sessions: {len(df)}")
    print(f"  Anomalous sessions: {len(anomalous_blocks)}")
    
    # Build transition graphs
    print("\n[2/4] Building transition graph...")
    
    all_transitions = defaultdict(int)
    anomalous_transitions = defaultdict(int)
    
    for idx, row in df.iterrows():
        event_ids = row['event_ids']
        block_id = row['BlockId']
        is_anomalous = block_id in anomalous_blocks
        
        # Extract transitions (consecutive pairs)
        for i in range(len(event_ids) - 1):
            edge = (event_ids[i], event_ids[i+1])
            all_transitions[edge] += 1
            
            if is_anomalous:
                anomalous_transitions[edge] += 1
    
    print(f"  Total unique transitions: {len(all_transitions)}")
    print(f"  Transitions in anomalous sessions: {len(anomalous_transitions)}")
    
    # Identify anomalous-only transitions
    anomalous_only = set()
    for edge in anomalous_transitions:
        # If this edge appears ONLY in anomalous sessions
        if edge in all_transitions:
            normal_count = all_transitions[edge] - anomalous_transitions[edge]
            if normal_count == 0:
                anomalous_only.add(edge)
    
    print(f"  Anomalous-only transitions: {len(anomalous_only)}")
    
    # Build NetworkX graph
    print("\n[3/4] Creating NetworkX graph...")
    
    G = nx.DiGraph()
    
    # Add edges (filter to top transitions to avoid clutter)
    top_edges = sorted(all_transitions.items(), key=lambda x: x[1], reverse=True)[:100]
    
    for (src, dst), count in top_edges:
        G.add_edge(src, dst, weight=count, anomalous=(src, dst) in anomalous_only)
    
    print(f"  Graph nodes: {len(G.nodes())}")
    print(f"  Graph edges: {len(G.edges())}")
    
    # Visualize
    print("\n[4/4] Generating visualization...")
    
    plt.figure(figsize=(16, 12))
    
    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=1337)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=800,
        alpha=0.9
    )
    
    # Draw edges (normal vs anomalous)
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d['anomalous']]
    anomalous_edges = [(u, v) for u, v, d in G.edges(data=True) if d['anomalous']]
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=normal_edges,
        edge_color='gray',
        alpha=0.5,
        arrows=True,
        arrowsize=15,
        width=1.5
    )
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=anomalous_edges,
        edge_color='red',
        alpha=0.8,
        arrows=True,
        arrowsize=15,
        width=2.5,
        style='dashed'
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=9,
        font_weight='bold'
    )
    
    plt.title('DeepLog Workflow Graph\n(Red dashed = anomalous-only transitions)', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('artifacts/visual/workflow_graph.png', dpi=200, bbox_inches='tight')
    print(f"✓ Saved workflow graph to artifacts/visual/workflow_graph.png")
    
    # Save graph statistics
    stats = {
        'total_nodes': len(G.nodes()),
        'total_edges': len(G.edges()),
        'normal_edges': len(normal_edges),
        'anomalous_edges': len(anomalous_edges),
        'top_anomalous_transitions': [
            {'from': src, 'to': dst, 'count': anomalous_transitions[(src, dst)]}
            for src, dst in list(anomalous_only)[:10]
        ]
    }
    
    with open('artifacts/visual/workflow_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved workflow stats to artifacts/visual/workflow_stats.json")
    
    return G, stats


if __name__ == '__main__':
    visualize_workflow()
    print("\n✓ Task Done: Workflow visualization completed")
    print("  - artifacts/visual/workflow_graph.png")
    print("  - artifacts/visual/workflow_stats.json")
