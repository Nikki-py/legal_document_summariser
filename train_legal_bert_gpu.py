"""
train_legal_bert_gpu.py -- standalone GPU training for the Legal Document Summariser (Day 3).

Fine-tunes Legal-BERT (nlpaueb/legal-bert-base-uncased) as a multi-label clause classifier on CUAD.
Needs nothing from Day 1/2: it downloads CUAD itself and contains its own copy of the rule layer.

    python train_legal_bert_gpu.py                 # train (resumes automatically if a checkpoint exists)
    python train_legal_bert_gpu.py --epochs 2       # fewer epochs if your GPU session is short

Output: legal_bert_trained.zip -- copy it to your own computer and unzip it INSIDE legal_doc_summariser/.
It contains models/legal_bert_clause_clf/ (the model Day 4 and Day 5 load) plus data/ (the clause
table and the train/val/test split Day 5 evaluates on).

If the session ends mid-training, keep the training_run/ folder and run the same command again:
it continues from the last finished epoch.
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from contextlib import nullcontext

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CUAD_JSON_URL = "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"

# ---------------- Legal-aware sentence splitter (Day 2, with the "al." bug fixed) ----------------

LEGAL_ABBREVIATIONS = [
    "Sec.", "Art.", "No.", "Inc.", "Ltd.", "Co.", "Corp.", "vs.", "viz.",
    "etc.", "i.e.", "e.g.", "U.S.", "U.S.C.", "Mr.", "Mrs.", "Ms.", "Dr.",
    "Fig.", "cf.", "Para.", "Cl.", "Esq.", "Jr.", "Sr.", "al.", "Ave.",
]
_ABBR_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(
    re.escape(a) for a in sorted(LEGAL_ABBREVIATIONS, key=len, reverse=True)) + ")")
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"“(\d])')


def split_into_legal_sentences(text):
    protected = _ABBR_RE.sub(lambda m: m.group(0).replace(".", "\u0000"), text)
    protected = re.sub(r"(?<=\d)\.(?=\d)", "\u0001", protected)
    pieces = SENTENCE_SPLIT_RE.split(protected)
    return [p.replace("\u0000", ".").replace("\u0001", ".").strip() for p in pieces if p.strip()]


# ---------------- Rule layer (copy of Day 2's RISK_PATTERNS, used for the ensemble table) ----------------

RISK_PATTERNS = {
    'Termination for Convenience': [
        r'terminate\s+this\s+agreement.{0,40}(sole discretion|for convenience|without cause|for any reason)',
        r'may terminate.{0,30}at any time.{0,30}without cause',
    ],
    'Automatic Renewal': [
        r'automatically renew', r'auto-renew', r'shall (automatically )?renew.{0,20}unless',
    ],
    'Indemnification': [
        r'indemnify', r'hold\s+harmless', r'defend.{0,20}indemnify',
    ],
    'Limitation of Liability': [
        r'limitation of liability', r'in no event shall.{0,60}liable',
        r'shall not exceed.{0,30}(fees|amount)',
    ],
    'Uncapped or Unlimited Liability': [
        r'no limitation on liability', r'unlimited liability', r'without limitation.{0,20}liability',
    ],
    'Non-Compete': [
        r'non[- ]compete', r'shall not,? (directly or indirectly,? )?compete',
        r'restrict\w* .{0,30}competing business',
    ],
    'Exclusivity': [
        r'exclusive(ly)? (right|license|arrangement|basis)', r'sole and exclusive',
    ],
    'Arbitration or Jury Waiver': [
        r'binding arbitration', r'waives?.{0,20}right to (a )?jury trial', r'arbitration association',
    ],
    'Assignment Restriction': [
        r'shall not assign', r'without (the )?prior written consent.{0,30}assign',
    ],
    'Confidentiality': [
        r'confidential information', r'non-disclosure',
    ],
    'Unilateral Modification': [
        r'sole discretion.{0,30}(modify|amend|change)', r'reserves the right to (change|modify|amend)',
    ],
    'Waiver of Rights': [
        r'waives?.{0,20}(any )?(right|claim)', r'irrevocably waives?',
    ],
    'Audit Rights': [
        r'right to audit', r'audit.{0,30}(records|books)',
    ],
    'IP Ownership Transfer': [
        r'shall own all right, title', r'work[- ]made[- ]for[- ]hire',
        r'assigns? all right, title and interest',
    ],
    'Insurance Requirement': [
        r'maintain insurance', r'certificate of insurance',
    ],
    'Liquidated Damages': [
        r'liquidated damages',
    ],
    'Most Favored Nation': [
        r'most favou?red nation',
    ],
    'Cap on Liability': [
        r'aggregate liability.{0,40}shall not exceed',
    ],
}

COMPILED_RISK_PATTERNS = {cat: [re.compile(p, re.IGNORECASE) for p in pats] for cat, pats in RISK_PATTERNS.items()}


def flag_clause(clause_text):
    return [cat for cat, pats in COMPILED_RISK_PATTERNS.items() if any(p.search(clause_text) for p in pats)]


RULE_TO_CUAD = {
    'Termination for Convenience': 'Termination For Convenience',
    'Automatic Renewal': 'Renewal Term',
    'Limitation of Liability': 'Cap On Liability',
    'Cap on Liability': 'Cap On Liability',
    'Uncapped or Unlimited Liability': 'Uncapped Liability',
    'Non-Compete': 'Non-Compete',
    'Exclusivity': 'Exclusivity',
    'Assignment Restriction': 'Anti-Assignment',
    'Audit Rights': 'Audit Rights',
    'IP Ownership Transfer': 'Ip Ownership Assignment',
    'Insurance Requirement': 'Insurance',
    'Liquidated Damages': 'Liquidated Damages',
    'Most Favored Nation': 'Most Favored Nation',
}


# ---------------- Data ----------------

def build_clause_df(json_path):
    """One row per clause with the CUAD categories whose answer span OVERLAPS the clause.

    Uses each answer's character offset (answer_start). The older version tested whether the first
    40 characters of an answer appeared anywhere in the clause, so a short answer such as a party
    name ("Company") labelled almost every clause in the contract as "Parties" (~43k false labels)."""
    with open(json_path) as f:
        cuad_raw = json.load(f)
    rows = []
    for entry in tqdm(cuad_raw['data'], desc='building clause table'):
        for para in entry['paragraphs']:
            context = para['context']
            spans = []  # (category, start, end)
            for qa in para['qas']:
                m = re.search(r'"([^"]+)"', qa['question'])
                cat = m.group(1) if m else qa['question'][:40]
                for a in qa.get('answers', []):
                    if a.get('text', '').strip():
                        spans.append((cat, a['answer_start'], a['answer_start'] + len(a['text'])))
            cursor = 0
            for clause in split_into_legal_sentences(context):
                start = context.find(clause, cursor)
                if start == -1:  # should not happen; fall back to a substring search from the beginning
                    start = context.find(clause)
                end = start + len(clause)
                if start != -1:
                    cursor = end
                labels = {cat for cat, s, e in spans if start != -1 and s < end and e > start}
                rows.append({'contract_id': entry['title'], 'clause_text': clause, 'labels': sorted(labels)})
    return pd.DataFrame(rows)


def load_data(data_dir, cuad_json=None):
    pkl = os.path.join(data_dir, 'cuad_clause_level_v2.pkl')
    if os.path.exists(pkl):
        print("Using existing", pkl)
        return pd.read_pickle(pkl)
    json_path = cuad_json or os.path.join(data_dir, 'CUAD_v1.json')
    if not os.path.exists(json_path):
        print("Downloading CUAD_v1.json (~40MB)...")
        r = requests.get(CUAD_JSON_URL, timeout=600)
        r.raise_for_status()
        with open(json_path, 'wb') as f:
            f.write(r.content)
    df = build_clause_df(json_path)
    df.to_pickle(pkl)
    return df


class ClauseDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def get_device(allow_cpu):
    if torch.cuda.is_available():
        print("Using NVIDIA GPU:", torch.cuda.get_device_name(0))
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device('mps')
    if allow_cpu:
        print("WARNING: no GPU found, running on CPU (--allow-cpu). This will be very slow.")
        return torch.device('cpu')
    sys.exit("No GPU found. Check `nvidia-smi` works, and that you installed the CUDA build of PyTorch "
             "(see README). Use --allow-cpu only for a quick test.")


def f2(p, r):
    return 5 * p * r / np.maximum(4 * p + r, 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-model', default='nlpaueb/legal-bert-base-uncased')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--batch-size', type=int, default=32, help='lower to 16 or 8 if you get CUDA out-of-memory')
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--n-categories', type=int, default=41, help='how many of the most frequent CUAD categories to learn (41 = all)')
    ap.add_argument('--neg-ratio', type=float, default=2.0)
    ap.add_argument('--work-dir', default='training_run', help='checkpoints + data live here (keep it to resume)')
    ap.add_argument('--output-zip', default='legal_bert_trained.zip')
    ap.add_argument('--cuad-json', default=None, help='path to CUAD_v1.json if you downloaded it by hand')
    ap.add_argument('--allow-cpu', action='store_true', help='for testing only')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = get_device(args.allow_cpu)
    use_amp = device.type == 'cuda'
    amp_ctx = (lambda: torch.autocast(device_type='cuda', dtype=torch.float16)) if use_amp else nullcontext

    W = args.work_dir
    data_dir = os.path.join(W, 'data')
    model_dir = os.path.join(W, 'models', 'legal_bert_clause_clf')
    ckpt_path = os.path.join(W, 'checkpoint.pt')
    os.makedirs(data_dir, exist_ok=True)

    # ---- data, labels, contract-level split ----
    clause_df = load_data(data_dir, args.cuad_json)
    clause_df = clause_df[clause_df['clause_text'].str.split().str.len() >= 4].reset_index(drop=True)
    print(f"{len(clause_df):,} clauses from {clause_df['contract_id'].nunique()} contracts")

    label_counts = Counter(l for ls in clause_df['labels'] for l in ls)
    categories = [c for c, _ in label_counts.most_common(args.n_categories)]
    cat2idx = {c: i for i, c in enumerate(categories)}

    def multi_hot(ls):
        v = np.zeros(len(categories), dtype=np.float32)
        for l in ls:
            if l in cat2idx:
                v[cat2idx[l]] = 1.0
        return v

    split_path = os.path.join(data_dir, 'split_contract_ids.json')
    if os.path.exists(split_path):
        split_ids = json.load(open(split_path))
    else:
        ids = np.array(sorted(clause_df['contract_id'].unique()))
        np.random.default_rng(args.seed).shuffle(ids)
        n_tr, n_va = int(0.70 * len(ids)), int(0.15 * len(ids))
        split_ids = {'train': ids[:n_tr].tolist(), 'val': ids[n_tr:n_tr + n_va].tolist(), 'test': ids[n_tr + n_va:].tolist()}
        json.dump(split_ids, open(split_path, 'w'))
    splits = {k: clause_df[clause_df['contract_id'].isin(v)].reset_index(drop=True) for k, v in split_ids.items()}
    for k in splits:
        print(f"  {k:5s}: {len(split_ids[k])} contracts, {len(splits[k]):,} clauses")

    train_df = splits['train']
    is_pos = train_df['labels'].apply(lambda ls: any(l in cat2idx for l in ls))
    neg = train_df[~is_pos].sample(n=min((~is_pos).sum(), int(args.neg_ratio * is_pos.sum())), random_state=args.seed)
    train_bal = pd.concat([train_df[is_pos], neg]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    Y_train = np.stack(train_bal['labels'].apply(multi_hot).values)
    pos_counts = Y_train.sum(axis=0)
    pos_weight = torch.tensor(np.clip((len(Y_train) - pos_counts) / np.maximum(pos_counts, 1), 1.0, 10.0), dtype=torch.float32)
    print(f"Training on {len(train_bal):,} clauses ({int(is_pos.sum()):,} labelled + {len(neg):,} unlabelled)")

    # ---- model ----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=args.max_len, padding=True, return_tensors='pt')
        enc['labels'] = torch.tensor(np.stack(labels), dtype=torch.float32)
        return enc

    def loader(df, bs, shuffle):
        labels = np.stack(df['labels'].apply(multi_hot).values)
        return DataLoader(ClauseDataset(df['clause_text'].tolist(), labels), batch_size=bs, shuffle=shuffle, collate_fn=collate)

    train_loader = loader(train_bal, args.batch_size, True)
    val_loader = loader(splits['val'], args.batch_size * 2, False)
    test_loader = loader(splits['test'], args.batch_size * 2, False)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(categories), problem_type='multi_label_classification',
        id2label=dict(enumerate(categories)), label2id=cat2idx).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(train_loader)
    warm = int(0.1 * total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    @torch.no_grad()
    def predict(dl):
        model.eval()
        P, Y = [], []
        for b in tqdm(dl, desc='predict', leave=False):
            Y.append(b.pop('labels').numpy())
            b = {k: v.to(device) for k, v in b.items()}
            with amp_ctx():
                P.append(torch.sigmoid(model(**b).logits.float()).cpu().numpy())
        return np.concatenate(P), np.concatenate(Y)

    def macro(y, p):
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average=None, zero_division=0)
        return pr.mean(), rc.mean(), f1.mean(), f2(pr, rc).mean()

    # ---- resume ----
    start_epoch, best_f2, history = 1, -1.0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get('categories') != categories or ck.get('base_model') != args.base_model:
            print("Found a checkpoint from a different setup (other categories/model) -- starting fresh.")
            ck = None
    else:
        ck = None
    if ck is not None:
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        start_epoch, best_f2, history = ck['epoch'] + 1, ck['best_f2'], ck['history']
        print(f"Resuming after epoch {ck['epoch']} (best val F2 so far {best_f2:.3f})")

    # ---- train ----
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        pbar = tqdm(train_loader, desc=f'epoch {epoch}/{args.epochs}')
        for step, b in enumerate(pbar, 1):
            y = b.pop('labels').to(device)
            b = {k: v.to(device) for k, v in b.items()}
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                logits = model(**b).logits
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            running += loss.item()
            if step % 20 == 0:
                pbar.set_postfix(loss=f'{running / step:.4f}')
        vp, vy = predict(val_loader)
        P, R, F1, F2 = macro(vy, (vp >= 0.5).astype(int))
        history.append({'epoch': epoch, 'train_loss': running / len(train_loader), 'val_precision': P,
                        'val_recall': R, 'val_f1': F1, 'val_f2': F2, 'minutes': (time.time() - t0) / 60})
        print(f"epoch {epoch}: loss {history[-1]['train_loss']:.4f} | val macro P/R/F1/F2 @0.5 "
              f"{P:.3f}/{R:.3f}/{F1:.3f}/{F2:.3f} | {history[-1]['minutes']:.1f} min")
        if F2 > best_f2:
            best_f2 = F2
            model.save_pretrained(model_dir); tokenizer.save_pretrained(model_dir)
            print("  -> best so far, saved")
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch, 'best_f2': best_f2, 'history': history,
                    'categories': categories, 'base_model': args.base_model}, ckpt_path)
        print(f"  checkpoint written ({ckpt_path}); safe to stop here and resume later")

    # ---- thresholds (validation), test evaluation ----
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    vp, vy = predict(val_loader)
    grid = np.round(np.arange(0.05, 0.51, 0.05), 2)
    thresholds = []
    for j in range(len(categories)):
        scores = []
        for t in grid:
            p, r, _, _ = precision_recall_fscore_support(vy[:, j], (vp[:, j] >= t).astype(int), average='binary', zero_division=0)
            scores.append(f2(np.array(p), np.array(r)))
        thresholds.append(float(grid[int(np.argmax(scores))]))
    thresholds = np.array(thresholds)

    tp, ty = predict(test_loader)
    pred = (tp >= thresholds).astype(int)
    p, r, f1, sup = precision_recall_fscore_support(ty, pred, average=None, zero_division=0)
    report = pd.DataFrame({'category': categories, 'threshold': thresholds, 'support': sup.astype(int),
                           'precision': p.round(3), 'recall': r.round(3), 'f1': f1.round(3)}).sort_values('recall')
    print("\nTest set (held-out contracts), tuned thresholds:\n" + report.to_string(index=False))
    print(f"Macro P/R/F1: {p.mean():.3f} / {r.mean():.3f} / {f1.mean():.3f}")

    test_df = splits['test']
    rule_flags = test_df['clause_text'].apply(flag_clause)
    lower = {c.lower(): c for c in label_counts}
    rule_to_cuad = {k: lower[v.lower()] for k, v in RULE_TO_CUAD.items() if v.lower() in lower}
    rows = []
    for cuad_cat in sorted(set(rule_to_cuad.values())):
        gold = test_df['labels'].apply(lambda ls: cuad_cat in ls).values
        if not gold.any():
            continue
        names = [k for k, v in rule_to_cuad.items() if v == cuad_cat]
        rule_hit = rule_flags.apply(lambda fl: any(n in fl for n in names)).values
        dl_hit = pred[:, cat2idx[cuad_cat]].astype(bool) if cuad_cat in cat2idx else np.zeros(len(gold), bool)
        rows.append({'CUAD category': cuad_cat, 'gold': int(gold.sum()),
                     'recall rules': round((rule_hit & gold).sum() / gold.sum(), 3),
                     'recall Legal-BERT': round((dl_hit & gold).sum() / gold.sum(), 3),
                     'recall ensemble': round(((rule_hit | dl_hit) & gold).sum() / gold.sum(), 3)})
    ens = pd.DataFrame(rows)
    print("\nRules vs Legal-BERT vs ensemble (test recall):\n" + ens.to_string(index=False))

    # ---- save + zip ----
    meta = {'base_model': args.base_model, 'categories': categories, 'thresholds': thresholds.tolist(),
            'max_len': args.max_len, 'rule_to_cuad': rule_to_cuad, 'epochs': args.epochs, 'lr': args.lr,
            'neg_ratio': args.neg_ratio, 'best_val_macro_f2_at_0.5': best_f2, 'device': device.type}
    json.dump(meta, open(os.path.join(model_dir, 'clf_meta.json'), 'w'), indent=2)
    models_root = os.path.join(W, 'models')
    pd.DataFrame(history).to_csv(os.path.join(models_root, 'training_history.csv'), index=False)
    report.to_csv(os.path.join(models_root, 'test_report_legal_bert.csv'), index=False)
    ens.to_csv(os.path.join(models_root, 'test_ensemble_recall.csv'), index=False)

    staging = os.path.join(W, '_zip')
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(models_root, os.path.join(staging, 'models'))
    os.makedirs(os.path.join(staging, 'data'))
    for f in ['cuad_clause_level_v2.pkl', 'split_contract_ids.json']:
        shutil.copy(os.path.join(data_dir, f), os.path.join(staging, 'data', f))
    out = shutil.make_archive(os.path.splitext(args.output_zip)[0], 'zip', staging)
    shutil.rmtree(staging)
    print(f"\nDONE. Copy {os.path.abspath(out)} to your own computer and unzip it inside legal_doc_summariser/.")


if __name__ == '__main__':
    main()
