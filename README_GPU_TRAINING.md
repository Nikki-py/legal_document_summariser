# GPU training: run this on the college computer

`train_legal_bert_gpu.py` (or `Train_LegalBERT_GPU.ipynb`) plus `requirements_training.txt` is everything needed to fine-tune the Legal-BERT clause classifier (Day 3) on a GPU machine.
It doesn't need Day 1 or Day 2 files: it downloads CUAD itself.

## 1. Check the GPU
Open a terminal (Command Prompt on Windows) and run:
```
nvidia-smi
```
You should see a table with the GPU name. If the command isn't found, that computer has no usable NVIDIA GPU (ask the lab).

## 2. Set up a fresh environment (one time)
Windows:
```
python -m venv lb_env
lb_env\Scripts\activate
```
Linux / macOS:
```
python3 -m venv lb_env
source lb_env/bin/activate
```
Then install the **CUDA build** of PyTorch. Get the exact command for your CUDA version from
https://pytorch.org/get-started/locally/ (pick Pip + your OS + CUDA). It looks like:
```
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
then:
```
pip install -r requirements_training.txt
```
Quick check (it should print `True` and the GPU name):
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> If you can't install anything on the lab computer, the same training also runs on Google Colab
> (Runtime -> Change runtime type -> T4 GPU). Upload this script, then run `!python train_legal_bert_gpu.py`
> in a cell, and download `legal_bert_trained.zip` at the end.

## 3. Train
```
python train_legal_bert_gpu.py
```
- Downloads CUAD (~40MB) and the Legal-BERT base model (~440MB) on the first run.
- Each epoch prints validation scores and writes a checkpoint. Default is 3 epochs.
- **Short on time?** Use `--epochs 2`. If you hit *CUDA out of memory*, add `--batch-size 16` (or 8).
- **Session ended before it finished?** Keep the `training_run/` folder and run the **same command** again
  later: it resumes after the last completed epoch. (If you can't keep the folder on that computer, copy
  `training_run/` to a USB stick and copy it back next time.)

## 4. Take the result home
When it prints `DONE`, copy **`legal_bert_trained.zip`** (a few hundred MB) to your own computer and unzip it
**inside your `legal_doc_summariser/` folder**. It adds:
- `models/legal_bert_clause_clf/`: the fine-tuned model (what Day 4 and Day 5 load)
- `models/*.csv`: training history and test results (put these in your report)
- `data/cuad_clause_level_v2.pkl` and `data/split_contract_ids.json`: the exact data and test split Day 5 evaluates on

Day 4 and Day 5 then run on your own laptop. They use a GPU if there is one, otherwise CPU.
