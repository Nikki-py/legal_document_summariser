"""
legal_pipeline.py
End-to-end Legal Document Summariser pipeline (written by Part 4 of Legal_Summariser_All_Days.ipynb, used by Part 5).

PDF -> text -> sections/clauses (rules) -> Legal-BERT clause flags
    -> section-wise abstractive summaries (BART) -> Markdown report.
"""

import os
import re
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

from legal_nlp_utils import extract_pdf_text, process_document, split_into_legal_sentences, to_cuad_names


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------- Category metadata ----------------

# "Key facts" categories (everything else is a red flag)
INFO_CATEGORIES = {
    'document name', 'parties', 'agreement date', 'effective date', 'expiration date',
    'governing law', 'warranty duration', 'license grant',
}

WHY_IT_MATTERS = {
    'termination for convenience': 'The other side can end the contract without any breach on your part.',
    'automatic renewal': 'The contract renews by itself unless someone cancels in time.',
    'renewal term': 'The contract renews by itself unless someone cancels in time.',
    'notice period to terminate renewal': 'Miss this notice window and you are locked in for another term.',
    'indemnification': 'You may have to pay the other side\'s losses or legal costs.',
    'limitation of liability': 'Limits how much you can recover if the other side causes you harm.',
    'cap on liability': 'Limits how much you can recover if the other side causes you harm.',
    'uncapped or unlimited liability': 'Your financial exposure has no upper limit.',
    'uncapped liability': 'Your financial exposure has no upper limit.',
    'non-compete': 'Restricts what work or business you can do, possibly after the contract ends.',
    'exclusivity': 'Stops you from working with or buying from anyone else.',
    'arbitration or jury waiver': 'You give up the right to go to court or to a jury trial.',
    'assignment restriction': 'You cannot transfer the contract (e.g. if you sell your business) without consent.',
    'anti-assignment': 'You cannot transfer the contract (e.g. if you sell your business) without consent.',
    'confidentiality': 'Limits what you can disclose, often for years after the contract ends.',
    'unilateral modification': 'The other side can change the terms without your agreement.',
    'waiver of rights': 'You give up legal rights or claims you would otherwise have.',
    'audit rights': 'The other side can inspect your books and records.',
    'ip ownership transfer': 'You hand over ownership of the work or IP you create.',
    'ip ownership assignment': 'You hand over ownership of the work or IP you create.',
    'insurance requirement': 'You must buy and keep specific insurance at your own cost.',
    'insurance': 'You must buy and keep specific insurance at your own cost.',
    'liquidated damages': 'A fixed penalty is payable on breach, regardless of the actual loss.',
    'most favored nation': 'You must give the other side terms at least as good as anyone else gets.',
    'change of control': 'A merger or sale of either company can trigger termination or consent rights.',
    'minimum commitment': 'You must buy or pay a minimum amount, even if you need less.',
    'revenue/profit sharing': 'Part of your revenue or profit goes to the other side.',
    'post-termination services': 'Obligations continue after the contract ends.',
    'covenant not to sue': 'You agree not to bring certain legal claims.',
    'no-solicit of employees': 'You cannot hire the other side\'s staff.',
    'no-solicit of customers': 'You cannot approach the other side\'s customers.',
    'non-disparagement': 'You cannot say negative things about the other side.',
    'rofr/rofo/rofn': 'The other side gets first refusal / first offer before you can deal elsewhere.',
    'price restrictions': 'Limits on how prices can change.',
    'volume restriction': 'Limits on how much you can buy, sell or use.',
    'third party beneficiary': 'Someone outside the contract can enforce it.',
    'irrevocable or perpetual license': 'A licence that cannot be withdrawn or never ends.',
    'joint ip ownership': 'IP is co-owned, which can complicate using or licensing it.',
    'source code escrow': 'Source code is held by a third party and released on certain events.',
}


def why_it_matters(category):
    return WHY_IT_MATTERS.get(category.lower(), 'Review this clause carefully before signing.')


def is_info_category(category):
    return category.lower() in INFO_CATEGORIES


# ---------------- Stage 2: Legal-BERT clause classifier ----------------

class ClauseClassifier:
    """Fine-tuned Legal-BERT multi-label clause classifier."""

    def __init__(self, model_dir, device=None):
        self.device = device or get_device()
        with open(os.path.join(model_dir, 'clf_meta.json')) as f:
            self.meta = json.load(f)
        self.categories = self.meta['categories']
        self.thresholds = np.array(self.meta['thresholds'])
        self.max_len = self.meta.get('max_len', 256)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        if self.device.type == 'cuda':
            self.model.half()

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=64):
        probs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(list(texts[i:i + batch_size]), truncation=True, max_length=self.max_len,
                                 padding=True, return_tensors='pt').to(self.device)
            probs.append(torch.sigmoid(self.model(**enc).logits.float()).cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros((0, len(self.categories)))

    def predict(self, texts, batch_size=64):
        """Returns (list of [(category, prob), ...] per text, prob matrix)."""
        probs = self.predict_proba(texts, batch_size)
        flags = [[(self.categories[j], float(p[j])) for j in np.where(p >= self.thresholds)[0]] for p in probs]
        return flags, probs


# ---------------- Stage 3: abstractive summariser ----------------

class Summarizer:
    """Seq2seq summariser loaded directly (transformers v5 removed pipeline('summarization'))."""

    def __init__(self, model_name='facebook/bart-large-cnn', device=None, max_input_tokens=900):
        self.device = device or get_device()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
        if self.device.type == 'cuda':
            self.model.half()
        # Avoid clashing with max_new_tokens/min_new_tokens
        self.model.generation_config.max_length = None
        self.model.generation_config.min_length = None
        self.max_input_tokens = max_input_tokens

    def n_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_text(self, text):
        """Split text into chunks of <= max_input_tokens, breaking only at sentence boundaries."""
        chunks, current, current_len = [], [], 0
        for sent in split_into_legal_sentences(text):
            n = self.n_tokens(sent)
            if current and current_len + n > self.max_input_tokens:
                chunks.append(' '.join(current))
                current, current_len = [], 0
            current.append(sent)
            current_len += n
        if current:
            chunks.append(' '.join(current))
        return chunks

    @torch.no_grad()
    def summarize_batch(self, texts, max_new_tokens=130, min_new_tokens=30, num_beams=4, batch_size=8):
        outputs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], truncation=True, max_length=1024,
                                 padding=True, return_tensors='pt').to(self.device)
            ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                      num_beams=num_beams, no_repeat_ngram_size=3, length_penalty=2.0,
                                      early_stopping=True)
            outputs.extend(self.tokenizer.batch_decode(ids, skip_special_tokens=True))
        return [o.strip() for o in outputs]

    def summarize(self, text, **kw):
        """Chunk -> summarise -> join (re-summarise if still long)."""
        if not text.strip():
            return ''
        chunks = self.chunk_text(text)
        parts = self.summarize_batch(chunks, **kw)
        joined = ' '.join(parts)
        if len(parts) > 1 and self.n_tokens(joined) > self.max_input_tokens // 2:
            return self.summarize(joined, **kw)
        return joined


# ---------------- Full pipeline ----------------

# Parties was learned from noisy labels: keep the model's Parties flag only in a document's first clauses
PARTIES_MAX_CLAUSE_INDEX = 5


def filter_parties(dl_flags, clause_index):
    """Drops the model's 'Parties' flag from clauses at position >= PARTIES_MAX_CLAUSE_INDEX in their document."""
    if clause_index < PARTIES_MAX_CLAUSE_INDEX:
        return dl_flags
    return [f for f in dl_flags if (f[0] if isinstance(f, tuple) else f) != 'Parties']


def analyse_text(raw_text, classifier, summarizer, min_section_words=40):
    """Runs Stages 1-3 on raw contract text and returns a dict with everything the report needs."""
    timings = {}
    t0 = time.time()
    df = process_document(raw_text)  # sections + legal sentence split + rule flags
    if df.empty:
        raise ValueError('No clauses found in the document text.')
    df['clause_text'] = df['clause_text'].str.replace(r'\s*\n\s*', ' ', regex=True)
    timings['structure_and_rules_s'] = time.time() - t0

    t0 = time.time()
    dl_flags, probs = classifier.predict(df['clause_text'].tolist())
    dl_flags = [filter_parties(f, i) for i, f in enumerate(dl_flags)]
    df['dl_flags'] = [[c for c, _ in f] for f in dl_flags]
    df['dl_conf'] = [{c: round(p, 3) for c, p in f} for f in dl_flags]
    # Rule names -> CUAD names, so a clause caught by both detectors is listed once
    df['rule_cats'] = df['rule_flags'].apply(to_cuad_names)
    df['all_flags'] = [sorted(set(r) | set(d)) for r, d in zip(df['rule_cats'], df['dl_flags'])]
    df['source'] = [
        'Both' if set(r) & set(d) else ('Rules + Legal-BERT' if r and d else ('Rules' if r else ('Legal-BERT' if d else '')))
        for r, d in zip(df['rule_cats'], df['dl_flags'])
    ]
    timings['classifier_s'] = time.time() - t0

    t0 = time.time()
    section_texts = df.groupby('section', sort=False)['clause_text'].apply(' '.join)
    long_sections = [(h, t) for h, t in section_texts.items() if len(t.split()) >= min_section_words]
    section_summaries = {}
    for heading, text in long_sections:
        section_summaries[heading] = summarizer.summarize(text, max_new_tokens=90, min_new_tokens=20)
    overview_source = ' '.join(section_summaries.values()) or ' '.join(section_texts.values)
    overview = summarizer.summarize(overview_source, max_new_tokens=180, min_new_tokens=60)
    timings['summarisation_s'] = time.time() - t0

    return {'clauses': df, 'section_summaries': section_summaries, 'overview': overview, 'timings': timings}


def build_report_markdown(result, title='Contract'):
    df = result['clauses']
    flagged = df[df['all_flags'].str.len() > 0]

    info_rows, risk_rows = [], []
    for _, row in flagged.iterrows():
        cats = row['all_flags']
        info = [c for c in cats if is_info_category(c)]
        risk = [c for c in cats if not is_info_category(c)]
        if info:
            info_rows.append((info, row))
        if risk:
            risk_rows.append((risk, row))

    lines = [f'# Contract Summary: {title}', '',
             '_Automatically generated. This is not legal advice: read every flagged clause in full before signing._', '',
             '## Overview', '', result['overview'] or '_(no summary produced)_', '']

    lines += ['## Key Details', '']
    if info_rows:
        seen = set()
        for cats, row in info_rows:
            key = row['clause_text'][:120]
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- **{', '.join(cats)}**: {row['clause_text']}")
    else:
        lines.append('_No key details (parties, dates, governing law) detected._')
    lines.append('')

    lines += [f'## ⚠ Red Flags ({len(risk_rows)} clauses)', '']
    if risk_rows:
        by_cat = {}
        for cats, row in risk_rows:
            for c in cats:
                by_cat.setdefault(c, []).append(row)
        for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
            lines.append(f'### {cat}')
            lines.append(f'_Why it matters:_ {why_it_matters(cat)}')
            lines.append('')
            for row in by_cat[cat]:
                conf = row['dl_conf'].get(cat)
                src = f"Legal-BERT {conf:.2f}" if conf is not None else 'Rules'
                if conf is not None and cat in row['rule_cats']:
                    src = f"Rules + Legal-BERT {conf:.2f}"
                lines.append(f"> {row['clause_text']}")
                lines.append(f">")
                lines.append(f"> _Section: {row['section']} · detected by: {src}_")
                lines.append('')
    else:
        lines.append('_No risky clauses detected by either the rule layer or the Legal-BERT classifier. '
                     'This does not guarantee the contract is safe: still read it in full._')
        lines.append('')

    lines += ['## Section-by-section Summary', '']
    for heading, summary in result['section_summaries'].items():
        lines += [f'**{heading}**: {summary}', '']

    n_flag = len(flagged)
    n_mixed = (df['source'] == 'Rules + Legal-BERT').sum()
    lines += ['---', f"_Scanned {len(df)} clauses in {df['section'].nunique()} sections; "
                     f"{n_flag} flagged ({(df['source'] == 'Rules').sum()} by rules only, "
                     f"{(df['source'] == 'Legal-BERT').sum()} by Legal-BERT only, "
                     f"{(df['source'] == 'Both').sum()} by both for the same category"
                     + (f", {n_mixed} by both for different categories" if n_mixed else '') + ")._"]
    return '\n'.join(lines)


def summarize_document(pdf_path, classifier, summarizer, out_dir=None):
    """One call, end to end: PDF path -> (markdown report, analysis dict). Saves the .md if out_dir is given."""
    raw_text = extract_pdf_text(pdf_path)
    result = analyse_text(raw_text, classifier, summarizer)
    title = os.path.splitext(os.path.basename(pdf_path))[0]
    report = build_report_markdown(result, title=title)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f'{title}_report.md'), 'w') as f:
            f.write(report)
    return report, result
