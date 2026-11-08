"""
make_test_contracts.py
Writes three synthetic contract PDFs (a mutual NDA, a residential lease, SaaS terms of service) into
sample_contracts/, plus synthetic_answer_key.json listing the clauses planted in each one and the categories
the pipeline should flag them with.

The contracts are fictional (invented companies and people). They exist so the whole pipeline -- PDF extraction,
section detection, sentence splitting, rules, Legal-BERT, report -- can be checked end to end with a known answer.
About a third of the planted clauses are deliberately paraphrased so keyword rules alone miss them.

    python make_test_contracts.py

Real contracts for the qualitative review can sit in the same folder; only files named synthetic_* are generated here.
"""

import json
import os

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'sample_contracts')

# Each contract: title, then sections of (heading, [paragraph, ...]).
# Planted clauses are written as (text, [acceptable categories], note); plain strings are filler.
# A planted clause counts as "caught" if its clause is flagged at all, and "right category" if one of the
# acceptable categories is among its flags. Category names are the ones the report uses (CUAD names, or the
# rule name when the rule has no CUAD equivalent).
BENIGN = 'benign'

CONTRACTS = {
    'synthetic_mutual_nda': {
        'title': 'MUTUAL NON-DISCLOSURE AGREEMENT',
        'sections': [
            (None, [
                ('This Mutual Non-Disclosure Agreement (the "Agreement") is entered into as of March 3, 2025 (the "Effective Date") '
                 'by and between Northwind Analytics Ltd., a company incorporated in England ("Northwind"), and Harbor Robotics Inc., '
                 'a Delaware corporation ("Harbor").', ['Parties', 'Agreement Date', 'Effective Date'], 'preamble: parties and dates'),
                'The parties wish to explore a possible collaboration on warehouse analytics (the "Purpose") and to protect the '
                'information they exchange for that Purpose.',
            ]),
            ('1. DEFINITIONS', [
                '"Confidential Information" means any non-public business, technical or financial information disclosed by either '
                'party, whether in writing, orally or by inspection, that is marked as confidential or would reasonably be understood '
                'to be confidential.',
            ]),
            ('2. OBLIGATIONS OF THE RECEIVING PARTY', [
                ('The Receiving Party shall hold all Confidential Information in strict confidence and shall not disclose it to any '
                 'third party for a period of five (5) years following the termination of this Agreement.',
                 ['Confidentiality'], 'confidentiality survives 5 years'),
                ('Pursuant to Sec. 2.1, the Receiving Party may share Confidential Information with its advisers who need to know it '
                 'for the Purpose, subject to prior approval. Each such adviser must be bound by written obligations at least as '
                 'protective as these.', ['Confidentiality'], 'splitter test: "Sec." and "approval." must not merge sentences'),
            ]),
            ('3. NON-SOLICITATION', [
                ('During the term of this Agreement and for twelve (12) months thereafter, neither party shall, directly or '
                 'indirectly, solicit for employment or hire any employee of the other party with whom it had contact in connection '
                 'with the Purpose.', ['No-Solicit Of Employees'], 'no-solicit of employees (rules only; no model head)'),
            ]),
            ('4. RESTRICTED ACTIVITIES', [
                ('For a period of two (2) years after the Effective Date, Harbor shall not engage in, or assist any third party in, '
                 'the development of any analytics platform that is substantially similar to the Northwind platform.',
                 ['Non-Compete'], 'non-compete, paraphrased (no "compete" keyword)'),
            ]),
            ('5. REMEDIES', [
                'Each party acknowledges that a breach of this Agreement may cause irreparable harm for which monetary damages '
                'would be an inadequate remedy.',
                ('In the event of any unauthorised disclosure, the breaching party shall pay the disclosing party the sum of '
                 'USD 250,000, which the parties agree is a genuine pre-estimate of loss and not a penalty.',
                 ['Liquidated Damages'], 'liquidated damages, paraphrased (no "liquidated damages" phrase)'),
                ('The Receiving Party shall indemnify and hold harmless the Disclosing Party against all losses arising from any '
                 'breach of this Agreement.', ['Indemnification'], 'indemnity'),
            ]),
            ('6. OWNERSHIP OF FEEDBACK', [
                ('All feedback, suggestions and improvements relating to the Disclosing Party\'s technology shall vest exclusively '
                 'in the Disclosing Party, and the Receiving Party hereby assigns all right, title and interest therein.',
                 ['Ip Ownership Assignment'], 'IP assignment of feedback'),
            ]),
            ('7. TERM AND TERMINATION', [
                ('This Agreement shall remain in force for three (3) years from the Effective Date.', ['Expiration Date'], 'term'),
                ('Northwind may terminate this Agreement at any time, for any reason, upon ten (10) days written notice to Harbor.',
                 ['Termination For Convenience'], 'one-sided termination for convenience'),
            ]),
            ('8. GENERAL', [
                ('Neither party may assign this Agreement without the prior written consent of the other party.',
                 ['Anti-Assignment'], 'anti-assignment'),
                ('Upon a change of control of Harbor, Northwind may terminate this Agreement immediately by written notice.',
                 ['Change Of Control'], 'change of control (rules only; no model head)'),
                ('This Agreement shall be governed by and construed in accordance with the laws of England and Wales.',
                 ['Governing Law'], 'governing law'),
                ('All notices under this Agreement shall be sent to the addresses set out above.', [BENIGN], 'benign: notices'),
                ('This Agreement may be executed in counterparts, each of which shall be deemed an original.', [BENIGN],
                 'benign: counterparts'),
            ]),
        ],
    },
    'synthetic_residential_lease': {
        'title': 'RESIDENTIAL LEASE AGREEMENT',
        'sections': [
            (None, [
                ('This Residential Lease Agreement (the "Lease") is made on June 1, 2025 between Maple Street Properties LLC '
                 '("Landlord") and Jordan Rivera ("Tenant") for the premises at 42 Elm Avenue, Apartment 3B, Springfield '
                 '(the "Premises").', ['Parties', 'Agreement Date', 'Effective Date'], 'preamble: parties and date'),
            ]),
            ('1. TERM', [
                ('The lease term begins on July 1, 2025 and ends on June 30, 2026.', ['Expiration Date', 'Effective Date'], 'term dates'),
                ('Thereafter, this Lease shall automatically renew for successive twelve-month terms unless Tenant delivers written '
                 'notice of non-renewal at least sixty (60) days before the end of the then-current term.',
                 ['Renewal Term', 'Notice Period To Terminate Renewal'], 'auto-renewal with notice window (rules only)'),
            ]),
            ('2. RENT', [
                ('Tenant shall pay monthly rent of $1,850 on the first day of each month by bank transfer.', [BENIGN], 'benign: rent'),
                ('Landlord may raise the rent at any time during the term upon thirty (30) days notice to Tenant.',
                 ['Unilateral Modification', 'Price Restrictions'], 'unilateral rent increase, paraphrased (hard case)'),
            ]),
            ('3. SECURITY DEPOSIT', [
                'Tenant shall pay a security deposit of $3,700 before moving in.',
                ('Tenant forfeits the entire security deposit if Tenant ends this Lease before the end of the term, and the parties '
                 'agree this amount constitutes liquidated damages and not a penalty.', ['Liquidated Damages'], 'deposit forfeiture'),
            ]),
            ('4. TERMINATION', [
                ('Landlord may terminate this Lease without cause by giving Tenant thirty (30) days written notice.',
                 ['Termination For Convenience'], 'landlord termination without cause'),
                ('Tenant may end this Lease early only by paying an early termination fee equal to two months\' rent.',
                 ['Liquidated Damages'], 'early termination fee'),
            ]),
            ('5. INSURANCE AND LIABILITY', [
                ('Tenant shall maintain renter\'s insurance with liability coverage of at least $100,000 and shall name Landlord as '
                 'an additional insured.', ['Insurance'], 'insurance requirement'),
                ('Landlord shall not be responsible for any damage to Tenant\'s property or injury to any person on the Premises, '
                 'however caused, including by Landlord\'s own negligence.', ['Cap On Liability', 'Limitation of Liability'],
                 'exculpatory clause, paraphrased (hard case)'),
                ('Tenant shall indemnify and hold Landlord harmless from all claims arising from Tenant\'s use of the Premises.',
                 ['Indemnification'], 'indemnity'),
                ('Nothing in this Lease shall limit or exclude Tenant\'s liability for damage to the Premises.',
                 ['Uncapped Liability'], 'uncapped tenant liability'),
            ]),
            ('6. USE OF THE PREMISES', [
                ('Tenant shall keep the Premises clean, sanitary and in good condition.', [BENIGN], 'benign: upkeep'),
                ('Smoking is not permitted anywhere inside the Premises.', [BENIGN], 'benign: smoking'),
                ('Landlord reserves the right to change the building rules at any time without notice to Tenant.',
                 ['Unilateral Modification'], 'unilateral rule changes'),
                ('Tenant shall not sublet the Premises or assign this Lease without Landlord\'s prior written consent.',
                 ['Anti-Assignment'], 'no subletting / assignment'),
            ]),
            ('7. DISPUTES', [
                ('Any dispute arising under this Lease shall be resolved by binding arbitration, and Tenant waives any right to a '
                 'jury trial.', ['Arbitration or Jury Waiver'], 'arbitration + jury waiver'),
                ('Tenant agrees not to post negative reviews or disparaging comments about Landlord or the Premises on any website.',
                 ['Non-Disparagement'], 'non-disparagement (rules only)'),
                ('This Lease shall be governed by the laws of the State of Illinois.', ['Governing Law'], 'governing law'),
            ]),
        ],
    },
    'synthetic_saas_terms': {
        'title': 'CLOUDLEDGER TERMS OF SERVICE',
        'sections': [
            (None, [
                ('These Terms of Service (the "Terms") form a binding agreement between CloudLedger Inc. ("CloudLedger") and the '
                 'business customer that accepts them ("Customer"), effective as of the date Customer first accesses the Service.',
                 ['Parties', 'Document Name', 'Effective Date'], 'preamble: parties'),
            ]),
            ('1. CHANGES TO THESE TERMS', [
                ('CloudLedger may modify these Terms at any time by posting the revised Terms on its website, and Customer\'s '
                 'continued use of the Service constitutes acceptance of the revised Terms.',
                 ['Unilateral Modification'], 'unilateral modification'),
            ]),
            ('2. SUBSCRIPTION AND FEES', [
                ('Each subscription renews for another billing period at CloudLedger\'s then-current rates unless Customer cancels '
                 'before the renewal date.', ['Renewal Term'], 'auto-renewal, paraphrased (no "automatically")'),
                ('Customer commits to a minimum annual spend of $25,000 regardless of actual usage.',
                 ['Minimum Commitment'], 'minimum commitment (model only)'),
                ('CloudLedger may increase subscription fees once per year by no more than seven percent (7%).',
                 ['Price Restrictions'], 'price increase cap'),
                ('CloudLedger will not charge Customer a higher price for the Service than it charges any other customer with a '
                 'comparable subscription.', ['Most Favored Nation'], 'most-favoured-customer pricing, paraphrased'),
                ('All fees are payable in US dollars within thirty (30) days of invoice.', [BENIGN], 'benign: payment terms'),
            ]),
            ('3. TERMINATION', [
                ('CloudLedger may suspend or terminate Customer\'s account at any time, with or without cause, and without prior '
                 'notice.', ['Termination For Convenience'], 'termination without cause'),
                ('Upon termination, CloudLedger will retain Customer data for thirty (30) days, after which it will be permanently '
                 'deleted.', ['Post-Termination Services'], 'post-termination data retention (model only)'),
            ]),
            ('4. INTELLECTUAL PROPERTY', [
                ('Customer grants CloudLedger a perpetual, irrevocable, worldwide, royalty-free license to use any feedback or '
                 'data Customer submits to improve CloudLedger\'s services.', ['Irrevocable Or Perpetual License', 'License Grant'],
                 'perpetual licence to customer data'),
            ]),
            ('5. LIMITATION OF LIABILITY', [
                ('IN NO EVENT SHALL CLOUDLEDGER\'S AGGREGATE LIABILITY EXCEED THE FEES PAID BY CUSTOMER IN THE TWELVE (12) MONTHS '
                 'PRECEDING THE CLAIM.', ['Cap On Liability'], 'liability cap'),
                ('CLOUDLEDGER SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES.', ['Cap On Liability'],
                 'consequential damages exclusion'),
                ('The foregoing limitations shall not apply to Customer\'s payment obligations or to Customer\'s indemnification '
                 'obligations.', ['Uncapped Liability'], 'carve-out: customer liability uncapped'),
                ('Customer shall defend, indemnify and hold harmless CloudLedger from any claims arising out of Customer\'s use of '
                 'the Service.', ['Indemnification'], 'indemnity'),
            ]),
            ('6. AUDIT', [
                ('CloudLedger may audit Customer\'s use of the Service, including Customer\'s books and records relating to user '
                 'counts, upon ten (10) days notice.', ['Audit Rights'], 'audit rights'),
            ]),
            ('7. GENERAL', [
                ('Customer may not assign these Terms without CloudLedger\'s prior written consent.', ['Anti-Assignment'],
                 'anti-assignment'),
                ('CloudLedger may assign these Terms without restriction, including in connection with a merger, acquisition or '
                 'sale of all or substantially all of its assets.', ['Change Of Control'], 'change of control / free assignment'),
                ('During the subscription term and for one year thereafter, Customer shall not solicit any employee of CloudLedger '
                 'to leave CloudLedger\'s employment.', ['No-Solicit Of Employees'], 'no-solicit of employees'),
                ('Any dispute shall be resolved by binding arbitration administered by the American Arbitration Association, and '
                 'Customer waives any right to participate in a class action.', ['Arbitration or Jury Waiver'],
                 'arbitration + class action waiver'),
                ('These Terms are governed by the laws of the State of California.', ['Governing Law'], 'governing law'),
                ('CloudLedger provides customer support by email during normal business hours.', [BENIGN], 'benign: support'),
                ('The Service is hosted in data centres located in the United States.', [BENIGN], 'benign: hosting'),
            ]),
        ],
    },
}

CSS = """
* {font-family: serif; font-size: 10.5pt; line-height: 1.35;}
h1 {font-size: 15pt; text-align: center; margin-bottom: 14pt;}
h2 {font-size: 11pt; margin-top: 12pt; margin-bottom: 4pt;}
p {margin-bottom: 6pt; text-align: justify;}
"""


def _html(contract):
    esc = lambda s: s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    parts = [f'<h1>{esc(contract["title"])}</h1>']
    for heading, paras in contract['sections']:
        if heading:
            parts.append(f'<h2>{esc(heading)}</h2>')
        # Each paragraph is a run of sentences, as in a real contract; planted clauses are single sentences.
        for p in paras:
            parts.append(f'<p>{esc(p[0] if isinstance(p, tuple) else p)}</p>')
    return '\n'.join(parts)


def write_pdf(contract, path):
    story = pymupdf.Story(html=_html(contract), user_css=CSS)
    page_rect = pymupdf.paper_rect('letter')
    body = page_rect + (72, 72, -72, -80)
    writer = pymupdf.DocumentWriter(path)
    more = True
    while more:
        dev = writer.begin_page(page_rect)
        more, _ = story.place(body)
        story.draw(dev)
        writer.end_page()
    writer.close()
    # Add "Page X of Y" footers, the most common PDF extraction artefact (cleaned by clean_extracted_text_light).
    doc = pymupdf.open(path)
    for i, page in enumerate(doc):
        page.insert_text((page_rect.width / 2 - 30, page_rect.height - 40), f'Page {i + 1} of {len(doc)}', fontsize=9)
    doc.saveIncr()
    doc.close()


def answer_key():
    key = {}
    for name, c in CONTRACTS.items():
        items = []
        for _, paras in c['sections']:
            for p in paras:
                if isinstance(p, tuple):
                    text, cats, note = p
                    items.append({'text': text, 'expected': [] if cats == [BENIGN] else cats,
                                  'benign': cats == [BENIGN], 'note': note})
        key[f'{name}.pdf'] = items
    return key


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, c in CONTRACTS.items():
        path = os.path.join(OUT_DIR, f'{name}.pdf')
        write_pdf(c, path)
        print('wrote', path)
    with open(os.path.join(OUT_DIR, 'synthetic_answer_key.json'), 'w') as f:
        json.dump(answer_key(), f, indent=1)
    print('wrote', os.path.join(OUT_DIR, 'synthetic_answer_key.json'))


if __name__ == '__main__':
    main()
