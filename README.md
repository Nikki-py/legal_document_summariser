# Legal Document Summariser

An NLP + deep learning course project. It reads a contract PDF and produces:

1. **Key Details**: parties, dates, governing law and similar facts.
2. **Red Flags**: risky clauses (auto-renewal, one-sided termination, liability caps, indemnities, non-competes, ...),
   quoted **verbatim** with a "why it matters" note and which detector caught them.
3. **A readable summary**: an abstractive overview plus a summary for each section.

Missing a risky clause costs more than over-flagging one, so the whole design is **recall-first**.

> Not legal advice.

## How it works

```
PDF ─► text (pdfplumber, PyMuPDF fallback)
    ─► sections (regex headings) + legal-aware sentence splitter + regex risk rules     [classical NLP]
    ─► fine-tuned Legal-BERT multi-label clause classifier                             [deep learning]
    ─► ensemble = rules OR Legal-BERT ─► Key Details + Red Flags (verbatim clauses)
    ─► section-wise BART summaries ─► overall summary                                  [deep learning]
    ─► Markdown report (+ Gradio demo)
```

- **Classifier:** `nlpaueb/legal-bert-base-uncased` fine-tuned on [CUAD v1](https://www.atticusprojectai.org/cuad)
  (510 contracts, 41 lawyer-labelled clause categories), split 70/15/15 by contract. Weighted BCE loss, negative
  down-sampling, per-category thresholds tuned for F2 on the validation contracts. Trained on a GPU.
- **Rules:** regex patterns for risk categories, mapped to CUAD names so they can be scored and merged with the model.
- **Summariser:** `facebook/bart-large-cnn` (GPU/MPS) or `sshleifer/distilbart-cnn-12-6` (CPU), inference only.

## Results (77 held-out test contracts)

| system (20 model categories) | macro P | macro R | macro F1 |
|---|---|---|---|
| Legal-BERT @0.5 | 0.510 | 0.800 | 0.596 |
| Legal-BERT, deployed | 0.492 | 0.812 | 0.594 |
| Ensemble (rules OR model) | 0.439 | 0.818 | 0.552 |

- Over all 35 evaluated categories, macro recall is 0.349 for rules, 0.464 for Legal-BERT and **0.705** for the ensemble.
- **80.3%** of labelled clauses in the test contracts are surfaced, while the reader only has to look at 21.8% of all clauses.
- Summaries on 50 BillSum documents: BART ROUGE-1/2/L **0.403 / 0.191 / 0.253** vs Lead-3 0.213 / 0.124 / 0.153.

Full results, error analysis and limitations: [`legal_doc_summariser/REPORT.md`](legal_doc_summariser/REPORT.md) and the
written report in [`report/`](report/).

## Repository layout

| Path | What it is |
|---|---|
| `Legal_Summariser_All_Days.ipynb` | The whole project in one notebook: data, rules, classifier, pipeline, evaluation. Runs on a laptop (CPU/MPS). |
| `Train_LegalBERT_GPU.ipynb`, `train_legal_bert_gpu.py` | Standalone, resumable GPU training used to train the model. See [`README_GPU_TRAINING.md`](README_GPU_TRAINING.md). |
| `legal_doc_summariser/legal_nlp_utils.py` | Sentence splitter, heading detection, regex risk rules. |
| `legal_doc_summariser/legal_pipeline.py` | PDF → Key Details, Red Flags, summaries, Markdown report. |
| `legal_doc_summariser/summarise_contract.py` | Command-line entry point. |
| `legal_doc_summariser/sample_contracts/` | Synthetic test contracts with an answer key. |
| `legal_doc_summariser/models/` | Model config, tokenizer and thresholds (weights not included, see below), training curves. |
| `report/` | Written project report (PDF/DOCX). |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Model weights

The fine-tuned weights (`model.safetensors`, 438 MB) are too big for a normal Git repo, so they are not committed.
Put the file in `legal_doc_summariser/models/legal_bert_clause_clf/` next to `config.json`, or retrain it with
`train_legal_bert_gpu.py` on a GPU. The CUAD data and caches are downloaded or rebuilt by the notebook.

## Usage

```bash
cd legal_doc_summariser
python summarise_contract.py contract.pdf            # report -> reports/<name>_report.md
python summarise_contract.py contract.pdf --no-summary   # rules + Legal-BERT only (fast)
python summarise_contract.py --selftest              # score the synthetic contracts
python summarise_contract.py --demo                  # Gradio web demo
```

## Limitations

- The model was trained before a labelling bug was fixed, so its "Parties" predictions are noisy; the pipeline only keeps
  them in a document's first few clauses.
- 4 risk categories (Renewal Term, Uncapped Liability, Liquidated Damages, Most Favored Nation) have no model head and rely on rules only.
- BART is trained on news, not contracts. Scanned PDFs need OCR. Clauses over 256 tokens are truncated.
