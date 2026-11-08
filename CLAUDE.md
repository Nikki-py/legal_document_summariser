# Legal Document Summariser (NLP + DL course project)

## Goal
Long legal contracts hide clauses a signer must know about: auto-renewal, one-sided termination, liability caps,
indemnities, non-competes and similar terms. The tool takes a contract PDF and produces:
1. **Key Details**: parties, dates, governing law and similar facts.
2. **Red Flags**: risky clauses, shown as the **original text** (never paraphrased), each with a "why it matters" note and which detector caught it.
3. **A readable summary**: an abstractive overview plus a summary for each section.

Missing a risky clause costs more than over-flagging one, so the design is **recall-first** throughout.

## Dataset
- **CUAD v1** (The Atticus Project): 510 commercial contracts, 41 clause categories labelled by lawyers, in SQuAD-style JSON
  (`data → paragraphs → context + qas`, where each answer has `text` and `answer_start`). The notebooks download it from
  `huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json`. `load_dataset("theatticusproject/cuad-qa")`
  no longer works because HF dropped script-based datasets.
- Converted to **one row per sentence/clause** with a multi-label list of CUAD categories. Current file:
  `data/cuad_clause_level_v2.pkl` (labels from character-offset overlap; see Known bugs #1).
  The old `cuad_clause_level.pkl` from the first GPU run has 114,522 clauses across 510 contracts.
- **Split by contract** 70/15/15, i.e. 357/76/77 contracts, saved in `data/split_contract_ids.json` so training and Day 5 evaluate the same test contracts.
- **BillSum** (`FiscalNote/billsum`, test split) for summarisation ROUGE, since CUAD has no reference summaries.
- A few real contracts (NDA, rental agreement, ToS) in `sample_contracts/` for qualitative testing.

## Model and approach
```
PDF ─► text (pdfplumber, PyMuPDF fallback)
    ─► sections (regex headings) + legal-aware sentence split + 18 regex rule flags   [classical NLP, Day 2]
    ─► fine-tuned Legal-BERT multi-label clause classifier                           [DL, trained on GPU]
    ─► ensemble = rules OR Legal-BERT  ─► Key Details + Red Flags (verbatim clauses)
    ─► section-wise BART summaries ─► overall summary                                  [DL, inference only]
    ─► Markdown report  (+ Gradio demo)
```
- **Classifier:** `nlpaueb/legal-bert-base-uncased` (110M params), fine-tuned end-to-end, sigmoid head, `BCEWithLogitsLoss` with
  `pos_weight` (capped at 10), unlabelled clauses down-sampled 2:1 in training only, AdamW lr 2e-5, 10% warm-up then linear decay,
  fp16 on CUDA, max 256 tokens, batch 32. The best epoch is chosen by validation macro-F2. **Per-category thresholds** (0.05–0.5)
  are tuned for F2 on the validation contracts.
- **Rules:** 18 regex categories (`legal_nlp_utils.RISK_PATTERNS`), mapped to CUAD names via `rule_to_cuad` for evaluation.
- **Summariser:** `facebook/bart-large-cnn` on GPU, `sshleifer/distilbart-cnn-12-6` on CPU. Loaded with `AutoModelForSeq2SeqLM`
  because transformers v5 removed `pipeline("summarization")`. Chunks are ≤900 tokens, split at sentence boundaries; a second
  pass runs if the joined summaries are still long.

## Decisions made (and why)
| Decision | Why |
|---|---|
| Local Jupyter instead of Colab Drive mounting | Drive mount failed with "credential propagation unsuccessful". Colab still works using upload/download zips. |
| Fresh venv instead of Anaconda base | conda and pip binary mismatches (pyarrow, numpy). |
| CUAD JSON parsed directly | HF script datasets are no longer supported. |
| Separate extractive red flags + abstractive summary | An abstractive model can silently drop a risky clause; red flags are always verbatim. |
| Rules OR model ensemble | Rules catch exact phrasing; the model catches paraphrases. The union maximises recall. |
| **Training must run on GPU** (nikki, 2026-10-07) | Course requires a deep model trained by us. Replaced the CPU plan (frozen MiniLM + logistic regression) with full Legal-BERT fine-tuning. |
| Standalone, resumable training (`train_legal_bert_gpu.py` / `Train_LegalBERT_GPU.ipynb`) | The college GPU computer is only available sometimes. A checkpoint is written each epoch and re-running resumes. Output is one `legal_bert_trained.zip`. |
| Day 4/5 run on CPU too | Only training needs the GPU; inference runs on the laptop. |
| F2 for model selection and thresholds | Weights recall twice as much as precision, matching the recall-first goal. |
| Plain PyTorch loop, not HF `Trainer` | Every step is visible for the report and it survives transformers API changes. |
| **Learn all 41 categories** (changed 2026-10-11, after the first run) | The first run learned only the top 20, so Renewal Term, Uncapped Liability, Liquidated Damages and Most Favored Nation had **0 model recall** and depended on rules that caught 0–30%. |

## Results so far (first GPU run, college computer, CUDA)
Settings: 3 epochs, top-20 categories, **old substring labels** (see Known bugs #1). About 8.7 min per epoch.

| epoch | train loss | val macro P | val macro R | val macro F1 | val macro F2 |
|---|---|---|---|---|---|
| 1 | 0.175 | 0.502 | 0.742 | 0.585 | 0.663 |
| 2 | 0.054 | 0.496 | 0.808 | 0.602 | 0.704 |
| 3 | 0.036 | 0.556 | 0.778 | 0.637 | **0.710** |

**Test (77 held-out contracts, tuned thresholds):** macro precision 0.530, recall **0.779**, F1 0.621.
Best: Governing Law (R 1.00, F1 0.95), Parties, Termination For Convenience (R 0.94), Insurance, Expiration Date, Anti-Assignment.
Weakest recall: Effective Date 0.45, Document Name 0.52, Post-Termination Services 0.64, Rofr/Rofo/Rofn 0.64.

**Rules vs Legal-BERT vs ensemble (test recall, 12 rule-mapped categories):** mean 0.106 / 0.559 / 0.596.
The rules alone are weak (0–30%), Legal-BERT does most of the work, and the ensemble adds a little (Cap On Liability 0.786→0.795,
Exclusivity 0.722→0.741). Four categories had no model head: Liquidated Damages 0.11, Most Favored Nation 0.00,
Renewal Term 0.30, Uncapped Liability 0.00, all from rules only.

## Earlier evaluation (Day 5 on the laptop, 2026-10-11; superseded by the 2026-10-15 re-run below)
Same first-run model, scored against the **corrected** labels (`cuad_clause_level_v2.pkl`), 77 test contracts, 15,004 clauses.
Outputs: `legal_doc_summariser/REPORT.md`, `eval_per_category.csv`, `eval_rouge.csv`, `figures/*.png`.

| system (20 model categories) | macro P | macro R | macro F1 |
|---|---|---|---|
| Legal-BERT @0.5 | 0.510 | 0.800 | 0.596 |
| Legal-BERT tuned | 0.471 | **0.822** | 0.568 |
| Ensemble | 0.461 | 0.823 | 0.559 |

- The 8 categories that have both a rule and a model head: rules R 0.107 / Legal-BERT 0.833 / ensemble 0.836.
- Weakest recall: Revenue/Profit Sharing 0.60, Rofr/Rofo/Rofn 0.63, Post-Termination Services 0.63. Clauses of 15 words or fewer are missed most (miss rate 0.39).
- Parties: support 157 (was 7,071 with the old labels), precision **0.018**. This is the noisy-label effect (Known bugs #5).
- Headline: 94.0% of the 1,524 labelled clauses are surfaced, but 62.3% of *all* clauses get flagged. Both numbers are inflated by
  raw Parties predictions, because Day 5 doesn't apply the `PARTIES_MAX_CLAUSE_INDEX` workaround (Known bugs #9).
- ROUGE on 50 BillSum docs: bart-large-cnn R-1/R-2/R-L 0.403/0.191/0.253 vs Lead-3 0.213/0.124/0.153 (39 s/doc on M1 MPS).

## Re-run on the laptop (2026-10-15, M1 MPS, headless via nbclient, outputs saved in the notebooks)
- **Day 1: OK** (37 s). `cuad_clause_level.pkl` is now rebuilt with **offset labels**: 116,497 clauses, 10,146 with ≥1 label,
  41 categories, Parties 913 (no longer 42,745). It replaces the old 114,522-clause substring file.
- **Day 2: OK** (16 min, mostly the rule check over the full table). Rule recall over 23 mapped CUAD categories on the full table: **0.537**
  (was 0.106 on 12 categories; rules were extended to cover more categories). Regenerated `legal_nlp_utils.py` is identical to the previous file.
- **Day 4: OK** (69 s). Regenerated `legal_pipeline.py` is identical. Synthetic contracts: NDA 13/19 clauses flagged, lease 16/20, SaaS ToS 18/22.
- **Day 5 dry run: OK** (4 test contracts, 2 ROUGE docs, 3 min). Synthetic answer key: 45/48 planted clauses caught (0.938), 43/48 right category, 1 false alarm.
- **Day 5 full run: OK** (2026-10-15, 38 min: Legal-BERT 445 s on 15,004 clauses, ROUGE 35.7 s/doc). **These supersede the 2026-10-11 Day 5 numbers.**
  It now evaluates 35 categories (20 model heads + 15 rules-only) and reports the **deployed** Parties filter, so Known bugs #9 and #10 are addressed.
  - 20 model categories, macro P/R/F1: @0.5 0.510/0.800/0.596; tuned 0.471/0.822/0.568; **deployed 0.492/0.812/0.594**; ensemble 0.439/0.818/0.552.
  - 8 categories with a rule and a model head, macro R: rules 0.487 / Legal-BERT 0.833 / ensemble **0.848**.
  - All 35 categories, macro R: rules 0.349 / Legal-BERT 0.464 / ensemble **0.705** (rules are what cover the 15 categories without a head).
  - Parties (157 gold): raw model flags 8,787 clauses at P 0.018 / R 0.987; deployed (first 5 clauses) 288 flags at P 0.431 / R 0.790.
  - Headline: **80.3%** of the 1,524 labelled clauses are surfaced (84.1% of those in a covered category); rules alone 35.5%, Legal-BERT alone 73.7%.
    Reading load **21.8%** of all clauses as deployed (63.8% with raw Parties flags).
  - Weakest: Price Restrictions 0 (no test support caught), Most Favored Nation 0/4, Covenant Not To Sue 0.25, No-Solicit Of Customers 0.43,
    Uncapped Liability 0.43 (all rules-only); model heads Revenue/Profit Sharing 0.60, Rofr/Rofo/Rofn 0.63, Post-Termination Services 0.63.
    Clauses of ≤15 words have the highest miss rate (0.47).
  - ROUGE, 50 BillSum docs: bart-large-cnn R-1/R-2/R-L **0.403/0.191/0.253** vs Lead-3 0.213/0.124/0.153.
  - Synthetic contracts: 45/48 planted clauses caught (0.938), 43/48 right category, 1 false alarm. Misses are paraphrased liquidated damages,
    exculpatory clause and MFN pricing; paraphrased auto-renewal and rent increase only got a Parties flag.
  - The interrupted first attempt was cut off by laptop sleep at ROUGE doc 36/50. Day 5 is now **resumable**:
  Legal-BERT test predictions are cached in blocks of 1,000 clauses in `data/day5_test_probs.npz` and each BillSum summary in `data/rouge_summaries_cache.json`,
  so a re-run skips finished work. Run it under `caffeinate -i` so the Mac doesn't sleep. Remaining cost: ~8 min Legal-BERT (if not cached) + ~30 s per uncached ROUGE doc.
- HF "unauthenticated requests" warnings are harmless here; all downloads succeeded.
- Gradio demo (`RUN_DEMO`) must be set False for headless runs, otherwise `launch()` can block.

## What's done
- Day 1: setup, CUAD download/parse, clause table, EDA, PDF ingestion.
- Day 2: legal-aware splitter, heading detection, 18-category rule layer, `legal_nlp_utils.py`.
- Training: standalone GPU script and notebook. **First run completed** on the college GPU (results above).
- Day 4: `legal_pipeline.py` with the full PDF → report pipeline.
- Day 5: evaluation (rules / model / ensemble, per category, "surfaced at all" headline), error analysis, BillSum ROUGE vs Lead-3,
  real-contract review sheets, Gradio demo, auto-generated `REPORT.md`.
- All code tested end-to-end with tiny stand-in models (the sandbox can't download the real ones).

## What's left
1. ✅ Complete `model.safetensors` received 2026-10-11 (438,014,016 bytes, loads and predicts correctly). The ready-to-use
   bundle is `legal_bert_trained.zip` in the project folder: unzip it inside `legal_doc_summariser/` (adds `models/legal_bert_clause_clf/`,
   the result CSVs, and `data/split_contract_ids.json`).
2. **No further GPU training planned** (decided 2026-10-11: no more GPU access). Use the first-run model as-is and state its
   limitations: noisy Parties training labels; 4 risk categories are rules-only. Day 5 rebuilds the **corrected** clause table
   (`cuad_clause_level_v2.pkl`) from `CUAD_v1.json` and evaluates the model against it, so the reported numbers are honest.
   If GPU access does come back (e.g. free Colab T4), re-run `Train_LegalBERT_GPU.ipynb` for the fixed labels and all 41 categories.
3. ✅ Day 4 and Day 5 ran on the laptop 2026-10-11 (Apple M1, 8 GB, MPS, so bart-large-cnn was used; outputs are saved in the notebooks).
   Day 5 takes about 40 min: Legal-BERT about 8 min at batch 16, ROUGE about 30 min for 50 docs. **Memory:** at batch 128, and with BART generating
   alongside Legal-BERT, the 8 GB Mac swapped until it stalled (twice). Day 5 now uses small batches off CUDA, frees memory before ROUGE and logs
   progress to `rouge_progress.log`. Close Safari/Notion etc. before running. `sample_contracts/` now has 3 synthetic contracts with an answer key (`make_test_contracts.py`); real contracts are still missing.
4. Add real contract PDFs (NDA, rental, ToS) to `legal_doc_summariser/sample_contracts/`, re-run Day 5 §5, then fill in the review sheets (`review_sheets/*.csv`) for the real contracts and re-run that Day 5 cell to get the real-world miss rate.
5. Run ROUGE with a larger `N_EVAL` for the final numbers. Write up the error analysis and limitations from Day 5's output.
6. Optional: improve the rule patterns for the weakest rules-only categories (MFN, Covenant Not To Sue, No-Solicit Of Customers, Uncapped Liability). Look at the missed examples Day 5 prints.

## Known bugs / limitations
1. **Fixed in code; the current model was trained before the fix: noisy "Parties" labels.** The old table labelled a clause with a category if the first 40
   characters of *any* answer appeared in it. Short answers like a party name ("Company") matched almost every sentence, giving
   **42,745 "Parties" labels (92% of all labels)**, 7,071 in the test set alone. This inflated Parties scores, made down-sampling
   ineffective (nearly every clause counted as "labelled"), and flooded Key Details. Labels now come from answer character
   offsets (Day 1, `train_legal_bert_gpu.py`, Day 3).
2. **Fixed: sentence splitter merged sentences.** The abbreviation "al." matched the end of "trial.", "approval.", "renewal.".
   Fixed in Day 2 / `legal_nlp_utils.py`, and `legal_pipeline.py` re-applies the fix if an old module is loaded.
3. **Fixed: headings like "1. SERVICES" weren't detected.** A pattern was added.
4. The trained model doesn't cover Renewal Term, Uncapped Liability, Liquidated Damages or Most Favored Nation (top-20 only), so only the weak rules catch them. The code now trains all 41 categories, but there is no GPU access to retrain.
5. Because of bug 1, the model predicts "Parties" for almost any clause naming a party (threshold 0.05). Workaround in
   `legal_pipeline.py`: model "Parties" flags are kept only in the first 5 clauses of a document (`PARTIES_MAX_CLAUSE_INDEX`).
   Day 5 still reports the raw Parties score, which will look poor against the corrected labels. Explain it in the report.
6. Rule layer recall is modest (0.537 over 23 mapped categories on the full table; 0.349 macro on the test set over 35 categories). "Indemnification" and "Confidentiality" rules have no CUAD category, so they can't be evaluated.
7. `bart-large-cnn` is trained on news, not contracts, which is a domain gap. Heading detection is template-dependent. Scanned PDFs need OCR. Clauses over 256 tokens are truncated.
8. Not legal advice.
9. **Fixed 2026-10-15:** Day 5's headline ("surfaced at all", reading load) and micro precision use raw model Parties flags, which fire on almost every
   clause. Re-compute them without Parties (or with the `PARTIES_MAX_CLAUSE_INDEX` workaround) for an honest headline.
10. **Fixed 2026-10-15 (now 35 categories incl. rules-only):** Day 5 evaluated only the model's 20 categories, so the 4 rules-only risk categories (Renewal Term, Uncapped Liability,
    Liquidated Damages, Most Favored Nation) don't appear in its tables. The first run's `test_ensemble_recall.csv` had them.
11. **Open:** the Day 4 report lists one clause under two headings, because rule names ("Limitation of Liability", "Assignment Restriction")
    aren't mapped to CUAD names ("Cap On Liability", "Anti-Assignment") before the union. `analyse_text` should apply `rule_to_cuad`
    from `clf_meta.json`. Until then, "Rules + Legal-BERT" attribution never appears and the footer's rules/BERT/both counts are wrong.
    Fix it in Day 4's `%%writefile` cell, not only in `legal_pipeline.py`, or re-running Day 4 overwrites the fix.

## Files (cleaned up 2026-10-30)
| File | Runs on | Purpose |
|---|---|---|
| `Legal_Summariser_All_Days.ipynb` | laptop (CPU/MPS ok) | **the whole project**: Part 1 data, Part 2 writes `legal_nlp_utils.py`, Part 3 classifier (`TRAIN_MODEL=False` loads the trained model), Part 4 writes `legal_pipeline.py` + runs sample contracts, Part 5 evaluation/ROUGE/`REPORT.md`. Simplified 2026-10-30 (127 → 103 cells, no duplicated code); full run ~3 min with the caches in `data/`, results identical to the 2026-10-15 numbers. |
| `Train_LegalBERT_GPU.ipynb` / `train_legal_bert_gpu.py` (+ `requirements_training.txt`, `README_GPU_TRAINING.md`) | **GPU computer** | the exact standalone scripts that trained the model (kept on purpose) |
| `legal_doc_summariser/summarise_contract.py` | laptop | CLI: report for a PDF, `--selftest`, `--demo` |
| `backups/` | | notebook before the cleanup, and the outputs before the re-run |
| `report/` | | the written report (docx/pdf) |
| `requirements.txt` | laptop | environment |

The old per-day notebooks, `_old_versions/`, `legal_bert_trained.zip` (duplicate of the unzipped model) and `legal_summariser_final_code.zip` were deleted 2026-10-30 with nikki's OK.

Folder layout on the laptop: `legal_doc_summariser/{data, models/legal_bert_clause_clf, sample_contracts, reports, review_sheets, figures}` next to the notebooks.
