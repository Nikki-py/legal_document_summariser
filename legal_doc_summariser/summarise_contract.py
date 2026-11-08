"""
summarise_contract.py
Command-line entry point for the Legal Document Summariser (no Jupyter needed).

    python summarise_contract.py contract.pdf [more.pdf ...]   # full report(s) -> reports/<name>_report.md
    python summarise_contract.py contract.pdf --no-summary      # rules + Legal-BERT only (fast, skips BART)
    python summarise_contract.py --selftest                     # score the synthetic contracts against their answer key
    python summarise_contract.py --demo                         # Gradio web demo

Run it from anywhere; paths are relative to this file. Needs models/legal_bert_clause_clf/ (unzip legal_bert_trained.zip
here). The summariser is downloaded from Hugging Face on first use.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from legal_pipeline import (ClauseClassifier, Summarizer, analyse_text, build_report_markdown,  # noqa: E402
                            extract_pdf_text, get_device)

CLF_DIR = os.path.join(HERE, 'models', 'legal_bert_clause_clf')
SAMPLE_DIR = os.path.join(HERE, 'sample_contracts')
REPORT_DIR = os.path.join(HERE, 'reports')


class NoSummary:
    """Stands in for the summariser when --no-summary is given: Key Details and Red Flags only."""

    def summarize(self, text, **kw):
        return ''


def load_models(args):
    device = torch.device(args.device) if args.device else get_device()
    t0 = time.time()
    classifier = ClauseClassifier(CLF_DIR, device=device)
    if args.no_summary:
        summarizer = NoSummary()
    else:
        name = args.summarizer or ('facebook/bart-large-cnn' if device.type != 'cpu' else 'sshleifer/distilbart-cnn-12-6')
        summarizer = Summarizer(name, device=device)
    print(f"Models loaded on {device} in {time.time() - t0:.0f}s"
          + ('' if args.no_summary else f" (summariser: {summarizer.model_name})"), file=sys.stderr)
    return classifier, summarizer, device


def run_one(path, classifier, summarizer, out_dir):
    t0 = time.time()
    result = analyse_text(extract_pdf_text(path), classifier, summarizer)
    title = os.path.splitext(os.path.basename(path))[0]
    report = build_report_markdown(result, title=title)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{title}_report.md')
    with open(out, 'w') as f:
        f.write(report)
    df = result['clauses']
    print(f"{os.path.basename(path)}: {len(df)} clauses, {int((df['source'] != '').sum())} flagged, "
          f"{time.time() - t0:.1f}s -> {out}", file=sys.stderr)
    return result, report


def selftest(classifier, summarizer, out_dir):
    from synthetic_check import load_answer_key, check_planted, summarise
    key = load_answer_key(SAMPLE_DIR)
    if not key:
        sys.exit("No synthetic contracts found. Run: python make_test_contracts.py")
    rows = []
    for name, items in key.items():
        result, _ = run_one(os.path.join(SAMPLE_DIR, name), classifier, summarizer, out_dir)
        r = check_planted(result, items)
        r.insert(0, 'contract', name)
        rows.append(r)
    per_item = pd.concat(rows, ignore_index=True)
    table = summarise(per_item)
    with pd.option_context('display.width', 250, 'display.max_colwidth', 60):
        print('\n' + table.to_string(index=False))
        missed = per_item[(~per_item['benign'] & ~per_item['right category']) | per_item['false alarm']]
        if len(missed):
            print('\nMissed, wrong category, or false alarm:')
            print(missed[['contract', 'note', 'expected', 'flags']].to_string(index=False))
    total = table.iloc[-1]
    ok = int(total['not found (splitting problem)']) == 0 and float(total['catch rate']) >= 0.8
    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'}: caught {int(total['caught'])}/{int(total['planted clauses'])} planted "
          f"clauses (threshold 80%), {int(total['not found (splitting problem)'])} lost by splitting.")
    return ok


def demo(classifier, summarizer, device):
    import gradio as gr

    def fn(pdf_file):
        if pdf_file is None:
            return 'Upload a PDF contract first.'
        path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
        t0 = time.time()
        result = analyse_text(extract_pdf_text(path), classifier, summarizer)
        report = build_report_markdown(result, title=os.path.splitext(os.path.basename(path))[0])
        return report + f"\n\n_Processed in {time.time() - t0:.1f}s on {device}._"

    gr.Interface(fn=fn, inputs=gr.File(label='Contract PDF', file_types=['.pdf'], type='filepath'),
                 outputs=gr.Markdown(label='Report'), title='Legal Document Summariser',
                 description='Key details, red-flag clauses (rules + fine-tuned Legal-BERT) and an abstractive summary.',
                 flagging_mode='never').launch()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pdfs', nargs='*', help='contract PDF(s) to summarise')
    ap.add_argument('--out', default=REPORT_DIR, help='folder for the Markdown reports (default: reports/)')
    ap.add_argument('--no-summary', action='store_true', help='skip the abstractive summary (much faster)')
    ap.add_argument('--summarizer', default=None, help='Hugging Face seq2seq model (default: bart-large-cnn on GPU, distilbart on CPU)')
    ap.add_argument('--device', default=None, choices=['cpu', 'cuda', 'mps'], help='default: best available')
    ap.add_argument('--selftest', action='store_true', help='run the synthetic contracts and score them against the answer key')
    ap.add_argument('--demo', action='store_true', help='launch the Gradio web demo')
    args = ap.parse_args()
    if not (args.pdfs or args.selftest or args.demo):
        ap.error('give at least one PDF, or --selftest, or --demo')
    if not os.path.exists(os.path.join(CLF_DIR, 'clf_meta.json')):
        sys.exit(f"Trained classifier not found in {CLF_DIR}. Unzip legal_bert_trained.zip inside {HERE}.")

    classifier, summarizer, device = load_models(args)
    ok = True
    for p in args.pdfs:
        _, report = run_one(p, classifier, summarizer, args.out)
        if len(args.pdfs) == 1:
            print(report)
    if args.selftest:
        ok = selftest(classifier, summarizer, args.out)
    if args.demo:
        demo(classifier, summarizer, device)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
