"""
legal_nlp_utils.py
Shared Stage-1 (classical NLP) utilities for the Legal Document Summariser project.
Written by Part 2 of Legal_Summariser_All_Days.ipynb and imported by the later parts.
"""

import json
import re

import pandas as pd
from tqdm.auto import tqdm

# ---------------- PDF ingestion ----------------

def extract_pdf_text(pdf_path):
    """PDF text with line breaks kept; pdfplumber, then PyMuPDF."""
    text = ''
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
    except Exception as e:
        print('pdfplumber failed:', e)
    if len(text.strip()) < 50:
        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
            doc = fitz.open(pdf_path)
            text = '\n'.join(page.get_text() for page in doc)
            doc.close()
        except Exception as e:
            print('PyMuPDF failed:', e)
    if len(text.strip()) < 50:
        raise ValueError(f'Could not extract text from {pdf_path} (scanned PDF? it would need OCR first).')
    return text


# ---------------- Legal-aware sentence splitting ----------------

LEGAL_ABBREVIATIONS = [
    "Sec.", "Art.", "No.", "Inc.", "Ltd.", "Co.", "Corp.", "vs.", "viz.",
    "etc.", "i.e.", "e.g.", "U.S.", "U.S.C.", "Mr.", "Mrs.", "Ms.", "Dr.",
    "Fig.", "cf.", "Para.", "Cl.", "Esq.", "Jr.", "Sr.", "al.", "Ave.",
]

# Abbreviations only match at the start of a word (so "renewal." isn't read as "al.")
_ABBR_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(
    re.escape(a) for a in sorted(LEGAL_ABBREVIATIONS, key=len, reverse=True)) + ")")

def _protect_periods(text):
    protected = _ABBR_RE.sub(lambda m: m.group(0).replace(".", "\u0000"), text)
    protected = re.sub(r"(?<=\d)\.(?=\d)", "\u0001", protected)
    return protected

def _restore_periods(text):
    return text.replace("\u0000", ".").replace("\u0001", ".")

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\u201c(\d])')

def split_into_legal_sentences(text):
    """Abbreviation- and numbering-aware sentence split."""
    protected = _protect_periods(text)
    pieces = SENTENCE_SPLIT_RE.split(protected)
    return [_restore_periods(p).strip() for p in pieces if p.strip()]

# ---------------- Document structure detection ----------------

def clean_extracted_text_light(text):
    """Removes page numbers and extra blank lines; keeps single line breaks."""
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.I)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

HEADING_PATTERNS = [
    re.compile(r"^\s*(ARTICLE|SECTION)\s+[IVXLCDM\d]+[\.:]?\s*(.*)$", re.IGNORECASE),
    re.compile(r"^\s*\d+(\.\d+)*\s+[A-Z][A-Za-z0-9 ,\-/&']{3,80}$"),
    re.compile(r"^\s*[A-Z][A-Z0-9 ,\-/&']{6,80}$"),
    re.compile(r"^\s*\d+(\.\d+)*\.\s+[A-Z][A-Za-z0-9 ,\-/&']{2,80}$"),
]

# A line ending in one of these words is a wrapped clause line, not a heading
_CONNECTOR_END = re.compile(r"\b(A|AN|AND|ANY|AS|AT|BE|BY|FOR|FROM|IN|IS|ITS|OF|ON|OR|SUCH|THAT|THE|TO|WITH)$", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.:;!?)\]\"\u201d]$")

def is_heading(line):
    line = line.strip()
    if not line or len(line) > 100:
        return False
    return any(pat.match(line) for pat in HEADING_PATTERNS)

def _looks_like_heading(line, prev_line, prev_was_heading):
    """is_heading plus context: a heading must start a new block."""
    if not is_heading(line):
        return False
    prev = prev_line.strip()
    if prev and not prev_was_heading and not _SENTENCE_END.search(prev):
        return False
    text = line.strip()
    if _CONNECTOR_END.search(text):  # "4.2 Late payments accrue interest at the rate of" is a wrapped clause
        return False
    if text.isupper() and len(text.split()) > 12 and not re.match(r"^\s*((ARTICLE|SECTION)\s|\d)", text, re.I):
        return False
    return True

def split_into_sections(raw_text):
    """Splits line-break-preserved text into {heading, text} sections."""
    lines = raw_text.split("\n")
    sections = []
    current_heading = "Preamble"
    current_lines = []
    prev_line, prev_was_heading = "", True
    for line in lines:
        if _looks_like_heading(line, prev_line, prev_was_heading):
            if current_lines:
                sections.append({"heading": current_heading, "text": "\n".join(current_lines).strip()})
            current_heading = line.strip()
            current_lines = []
            prev_was_heading = True
        else:
            current_lines.append(line)
            prev_was_heading = False if line.strip() else prev_was_heading
        if line.strip():
            prev_line = line
    if current_lines:
        sections.append({"heading": current_heading, "text": "\n".join(current_lines).strip()})
    return [s for s in sections if s["text"]]

# ---------------- Rule-based risk flagging ----------------

# Developed on CUAD train contracts, checked on validation
RISK_PATTERNS = {
    'Termination for Convenience': [
        r'terminate\s+this\s+agreement.{0,40}(sole discretion|for convenience|without cause|for any reason)',
        r'may terminate.{0,30}at any time.{0,30}without cause',
        r'terminat\w*.{0,60}(with or without cause|without cause|for convenience|for any reason or no reason|for any or no reason)',
        r'may terminate this agreement.{0,40}at any time',
        r'until terminated by either party',
        r'(either|each) party may terminate this agreement.{0,40}(upon|by giving|on|with).{0,40}notice',
    ],
    'Automatic Renewal': [
        r'automatically (be )?(renew|extend)', r'(renew|extend)(ed|s)? automatically', r'auto-?renew',
        r'shall (automatically )?renew.{0,20}unless', r'successive (renewal )?(terms?|periods?)',
        r'(additional|successive|further) (\w+ )?(\(\d+\) )?(-?year|-?month|year|month)s? (renewal )?(terms?|periods?)',
        r'(option|right) to (renew|extend)', r'renewal (term|period)', r'month[- ]to[- ]month', r'year[- ]to[- ]year',
    ],
    'Notice Period to Avoid Renewal': [
        r'(notice|notify|notification|notifies).{0,80}(non-?renewal|not to renew|intent\w* not to (renew|extend)|not to extend)',
        r'unless.{0,100}notice.{0,100}(prior to|before|in advance of).{0,40}(end|expir\w*) of the (then[- ]current|initial|renewal|current)',
        r'(notice|notif\w*).{0,80}(no later than|at least|not less than|prior to|before).{0,60}(end|expir\w*) of (the|any|such) (then[- ]current |then in effect |initial |renewal |relevant |current )?(term|period)',
        r'until terminated by either party.{0,40}notice',
        r'unless.{0,60}party.{0,60}notice.{0,60}(intent|intention|election) (to terminate|not to)',
    ],
    'Indemnification': [
        r'indemnif', r'hold\s+harmless', r'defend.{0,20}indemnify',
    ],
    'Limitation of Liability': [
        r'limitation of liability', r'in no event (shall|will).{0,100}(liable|liability)',
        r'shall not exceed.{0,30}(fees|amount)',
        r'(liability|liable).{0,120}(shall not exceed|will not exceed|not to exceed|limited to|in excess of|greater than)',
        r'(neither party|no party).{0,60}liable.{0,80}(indirect|consequential|incidental|special|punitive)',
    ],
    'Uncapped or Unlimited Liability': [
        r'no limitation on liability', r'unlimited liability', r'without limitation.{0,20}liability',
        r'(limitations?|exclusions?|caps?) (of|on) liability.{0,60}(shall|will) not apply',
        r'(foregoing|above|preceding|these) (limitations?|exclusions?).{0,40}(shall|will|do) not apply',
        r'(shall|will) not (apply|limit).{0,80}(indemnif|gross negligence|willful misconduct|fraud|breach of (section|article|confidential))',
        r'except (for|with respect to|in connection with|as provided|to the extent).{0,250}(in no event|shall not be liable|will not be liable|(neither|no) party (shall|will) be liable)',
        r'^\W*(\w+ ){0,3}(except|excluding|other than)\b.{0,400}(liab|damages)',
        r'nothing (in|herein).{0,80}(limit|exclude|restrict).{0,40}liabilit',
        r'liab\w*.{0,200}\((except|excluding|other than)',
    ],
    'Non-Compete': [
        r'non[- ]?compet', r'shall not,? (directly or indirectly,? )?compete',
        r'restrict\w* .{0,30}competing business',
        r'competing (products?|business(es)?|services?|technolog)', r'compet(itive|ing) with',
        r'(not|neither).{0,80}(engage|participate|invest|represent|develop|manufacture|sell|market|distribute|promote|commercializ)\w*.{0,80}(compet|similar)',
    ],
    'Exclusivity': [
        r'exclusive(ly)? (right|license|arrangement|basis|distribut|supplier|provider|dealing|sales)', r'sole and exclusive',
        r'(exclusively|solely) (from|to|through|with)', r'(not|neither).{0,60}(purchase|buy|obtain|source).{0,40}from any (other|third)',
    ],
    'Arbitration or Jury Waiver': [
        r'binding arbitration', r'waives?.{0,20}right to (a )?(trial by )?jury', r'arbitration association',
        r'trial by jury', r'class action',
    ],
    'Assignment Restriction': [
        r'shall not assign', r'without (the )?prior written consent.{0,30}assign',
        r'(may|shall|will|can) not (\w+ ){0,4}(assign|transfer)', r'not (be )?(assignable|transferable)',
        r'assign\w*.{0,100}without.{0,30}(prior )?(express )?(written )?consent',
        r'(no|neither) party (may|shall|will).{0,30}assign',
    ],
    'Change of Control': [
        r'change (of|in) (control|ownership)', r'change of management',
        r'(merger|consolidation|acquisition|reorganization).{0,100}(assign|terminat|consent|transfer)',
        r'(sale|transfer) of (all or )?substantially all', r'(liquidity|liquidation) event',
        r'(sale|transfer|acquisition) of.{0,40}(voting|controlling|equity) (stock|interests?|shares|securities)',
    ],
    'Confidentiality': [
        r'confidential information', r'non-disclosure',
    ],
    'Unilateral Modification': [
        r'sole discretion.{0,30}(modify|amend|change)', r'reserves the right to (change|modify|amend)',
        r'may (change|modify|amend|update) (these|this|the) (terms|agreement).{0,40}(at any time|from time to time|without)',
    ],
    'Waiver of Rights': [
        r'waives?.{0,20}(any )?(right|claim)', r'irrevocably waives?',
    ],
    'Covenant Not to Sue': [
        r'covenants? not to (sue|assert|bring)',
        r'(not|never)[\w ,]{0,60}(contest|challenge|attack|dispute|oppose)\w*.{0,80}(validity|ownership|title|rights|enforceability)',
        r'(not|never)[\w ,]{0,40}(commence|bring|institute|assert|file)\w*.{0,60}(suit|action|claim|proceeding)',
        r'(not|never)[\w ,()]{0,80}(oppose|challenge|contest|attack)\b',
        r'(not|never).{0,80}(impair|jeopardi[sz]e).{0,40}(rights?|title|ownership|marks?)',
        r'releases?( and discharges?)?.{0,60}(from )?(all|any) (and all )?claims', r'general release',
    ],
    'No-Solicit of Employees': [
        r'non-?solicit',
        r'solicit\w*.{0,80}(employ|personnel|staff|consultants?|contractors?)',
        r'(hire|employ|engage|recruit)\w*.{0,50}(employees?|personnel|staff) of',
        r'(induce|entice|encourage)\w*.{0,60}(employee|personnel|staff).{0,60}(leave|terminat)',
    ],
    'No-Solicit of Customers': [
        r'solicit\w*.{0,80}(customers?|clients?|suppliers?|vendors?|accounts?|business)',
        r'divert\w*.{0,60}(business|customers?|clients?)', r'circumvent',
        r'interfer\w*.{0,60}relationships?.{0,60}(customers?|clients?|suppliers?)',
    ],
    'Non-Disparagement': [
        r'disparag', r'defam', r'(negative|derogatory|critical).{0,30}(comments?|statements?|remarks?)',
        r'(harm|damage|injur|impair|jeopardi)\w*.{0,40}(reputation|goodwill|image)', r'publicly criticiz',
    ],
    'Audit Rights': [
        r'right to audit', r'audit.{0,30}(records|books)', r'\baudit', r'(inspect|examin)\w*.{0,60}(books|records|accounts)',
    ],
    'IP Ownership Transfer': [
        r'shall own all right, title', r'work[- ]made[- ]for[- ]hire',
        r'assigns? all right, title and interest',
        r'(hereby )?assign\w*.{0,60}(right,? title,? and interest|intellectual property)',
        r'(shall|will) (vest|belong) (solely |exclusively )?(in|to)',
        r'(intellectual property|inventions?|work product|deliverables|improvements).{0,80}(shall|will) (be|become|remain) the (sole and exclusive |exclusive |sole )?property',
    ],
    'Joint IP Ownership': [
        r'joint(ly)?[- ]own', r'joint (intellectual property|inventions?|ip|patents?|technology)', r'undivided interest', r'co-?own',
    ],
    'Irrevocable or Perpetual License': [
        r'(perpetual|irrevocable)[\w ,\-()]{0,80}licen[cs]e', r'licen[cs]e[\w ,\-()]{0,60}(perpetual|irrevocable)',
    ],
    'Source Code Escrow': [
        r'escrow', r'source code',
    ],
    'Insurance Requirement': [
        r'maintain insurance', r'certificate of insurance', r'insurance', r'\binsured\b',
    ],
    'Liquidated Damages': [
        r'liquidated damages', r'termination (fee|charge|payment)', r'break-?up fee', r'early termination (fee|charge|payment)',
        r'(as|constitute)( a)? (penalty|forfeiture)',
    ],
    'Most Favored Nation': [
        r'most favou?red (nation|customer|pricing|licensee)', r'(no|not) less favou?rabl[ey]',
        r'more favou?rable (terms|prices?|pricing|conditions|rates?)', r'(lowest|best) (net )?prices?',
        r'(not|never) (be )?charged? (a )?higher',
    ],
    'Price Restrictions': [
        r'(increase|raise|adjust)\w*.{0,60}(prices?|fees?|rates?).{0,80}(no more than|not more than|not to exceed|shall not exceed|once (per|in|each|a)|annually|cpi)',
        r'(prices?|fees?|rates?).{0,40}(shall|will|may) not (be )?(increased?|exceed|changed?)',
        r'consumer price index', r'\bCPI\b',
    ],
    'Cap on Liability': [
        r'aggregate liability.{0,40}shall not exceed',
    ],
    'Third Party Beneficiary': [
        r'third[- ]party beneficiar', r'intended beneficiar',
    ],
}

# Rule category -> CUAD category (unmapped rules keep their own name)
RULE_TO_CUAD = {
    'Termination for Convenience': 'Termination For Convenience',
    'Automatic Renewal': 'Renewal Term',
    'Notice Period to Avoid Renewal': 'Notice Period To Terminate Renewal',
    'Limitation of Liability': 'Cap On Liability',
    'Cap on Liability': 'Cap On Liability',
    'Uncapped or Unlimited Liability': 'Uncapped Liability',
    'Non-Compete': 'Non-Compete',
    'Exclusivity': 'Exclusivity',
    'Assignment Restriction': 'Anti-Assignment',
    'Change of Control': 'Change Of Control',
    'Covenant Not to Sue': 'Covenant Not To Sue',
    'No-Solicit of Employees': 'No-Solicit Of Employees',
    'No-Solicit of Customers': 'No-Solicit Of Customers',
    'Non-Disparagement': 'Non-Disparagement',
    'Audit Rights': 'Audit Rights',
    'IP Ownership Transfer': 'Ip Ownership Assignment',
    'Joint IP Ownership': 'Joint Ip Ownership',
    'Irrevocable or Perpetual License': 'Irrevocable Or Perpetual License',
    'Source Code Escrow': 'Source Code Escrow',
    'Insurance Requirement': 'Insurance',
    'Liquidated Damages': 'Liquidated Damages',
    'Most Favored Nation': 'Most Favored Nation',
    'Price Restrictions': 'Price Restrictions',
    'Third Party Beneficiary': 'Third Party Beneficiary',
}

COMPILED_RISK_PATTERNS = {cat: [re.compile(p, re.IGNORECASE) for p in pats] for cat, pats in RISK_PATTERNS.items()}

def flag_clause(clause_text):
    """Returns the list of rule categories whose patterns match this clause."""
    matches = []
    for category, patterns in COMPILED_RISK_PATTERNS.items():
        if any(pat.search(clause_text) for pat in patterns):
            matches.append(category)
    return matches

def to_cuad_names(rule_flags):
    """Maps rule category names to CUAD names (unmapped rule categories keep their name), without duplicates."""
    out = []
    for r in rule_flags:
        c = RULE_TO_CUAD.get(r, r)
        if c not in out:
            out.append(c)
    return out

# ---------------- Combined Stage-1 pipeline ----------------

def process_document(raw_text):
    """Full Stage-1 pipeline: detect sections, split into clauses, apply rule-based flags."""
    light_cleaned = clean_extracted_text_light(raw_text)
    sections = split_into_sections(light_cleaned)
    rows = []
    for sec in sections:
        clauses = split_into_legal_sentences(sec['text'])
        for clause in clauses:
            # Join PDF line wraps so patterns can match across them
            clause = re.sub(r"\s*\n\s*", " ", clause)
            if len(clause.split()) < 4:
                continue  # skip near-empty fragments (stray headers, page artifacts)
            rows.append({
                'section': sec['heading'],
                'clause_text': clause,
                'rule_flags': flag_clause(clause),
            })
    return pd.DataFrame(rows)

# ---------------- CUAD clause table (training and evaluation) ----------------

def build_clause_df(json_path):
    """One row per clause, labelled with the CUAD categories whose answer span overlaps it."""
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
