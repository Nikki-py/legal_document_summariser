"""
synthetic_check.py
Scores the pipeline's output on the synthetic contracts against synthetic_answer_key.json (see make_test_contracts.py).

For each planted clause:
  caught          the clause it ends up in was flagged at all (shown in Key Details or Red Flags)
  right category  one of the expected categories is among that clause's flags
Benign filler clauses that get flagged are counted as false alarms.
"""

import json
import os
import re

import pandas as pd


def _norm(s):
    return re.sub(r'\s+', ' ', s).strip().lower()


def load_answer_key(sample_dir):
    path = os.path.join(sample_dir, 'synthetic_answer_key.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def check_planted(result, items):
    """result: analyse_text() output for one contract. items: its answer-key entries. Returns one row per item."""
    df = result['clauses']
    clauses = [_norm(t) for t in df['clause_text']]
    rows = []
    for it in items:
        probe = _norm(it['text'])[:60]  # opening words; a planted item may span two sentences
        idx = next((k for k, c in enumerate(clauses) if probe in c), None)
        flags = list(df['all_flags'].iloc[idx]) if idx is not None else []
        source = df['source'].iloc[idx] if idx is not None else ''
        rows.append({
            'note': it['note'],
            'expected': ', '.join(it['expected']) or '(none: benign)',
            'found clause': idx is not None,
            'flags': ', '.join(flags),
            'source': source,
            'benign': it['benign'],
            'caught': bool(flags) and not it['benign'],
            'right category': bool(set(flags) & set(it['expected'])),
            'false alarm': bool(flags) and it['benign'],
        })
    return pd.DataFrame(rows)


def summarise(per_item):
    """per_item: concatenated check_planted rows with a 'contract' column. Returns a one-row-per-contract table."""
    out = []
    for name, g in per_item.groupby('contract', sort=False):
        planted, benign = g[~g['benign']], g[g['benign']]
        out.append({'contract': name, 'planted clauses': len(planted),
                    'caught': int(planted['caught'].sum()), 'right category': int(planted['right category'].sum()),
                    'benign clauses': len(benign), 'false alarms': int(benign['false alarm'].sum()),
                    'not found (splitting problem)': int((~g['found clause']).sum())})
    t = pd.DataFrame(out)
    if len(t):
        total = t.drop(columns='contract').sum()
        total['contract'] = 'TOTAL'
        t = pd.concat([t, total.to_frame().T], ignore_index=True)
        t['catch rate'] = (t['caught'] / t['planted clauses']).astype(float).round(3)
        t['right-category rate'] = (t['right category'] / t['planted clauses']).astype(float).round(3)
    return t
