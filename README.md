# TFDR

**T**raditional **F**ormula-**D**isease **R**elationships (TFDR) is a manually annotated corpus, designed for the extraction of relationship between traditional formula (TF) and disease from abstracts in PubMed. The TFDR corpus comprises a total of 6,211 TF mentions and 7,166 disease mentions, along with 1,103 relationships between them, all of which are contained within 740 abstracts. To the best of our knowledge, the TFDR corpus represents the first of its kind.


## Statistics

<table>
<tbody>
  <tr>
    <td rowspan="2"><b>Abstracts</b></td>
    <td colspan="2"><b>Traditional formula</b></td>
    <td colspan="2"><b>Disease</b></td>
  </tr>
  <tr>
    <td><b>Entities</b></td>
    <td><b>Unique entities</b></td>
    <td><b>Entities</b></td>
    <td><b>Unique entities</b></td>
  </tr>
  <tr>
    <td>740</td>
    <td>6,211</td>
    <td>201</td>
    <td>7,166</td>
    <td>694</td>
  </tr>
  <tr>
    <td><b>Key-sentence</b></td>
    <td colspan="4"><b>Relation</b></td>
  </tr>
  <tr>
    <td rowspan="2">744</td>
    <td><b>Treatment of Disease</b></td>
    <td><b>Association</b></td>
    <td><b>Cause of Side-effect</b></td>
    <td><b>Negative</b></td>
  </tr>
  <tr>
    <td>919</td>
    <td>141</td>
    <td>30</td>
    <td>13</td>
  </tr>
</tbody>
</table>

## Corpus
The raw data format follows the PubTator data format extension, with each document separated by a new line. It comprises three sections for each document: Text, NER (Named Entity Recognition), and RE (Relation Extraction) information, as demonstrated below: 

**[Image inserts]**

Processed files are created from raw data, comprising labels and texts devoid of headers. The columns within the file are separated by tab delimiters.


## Codes
Python codes for key-sentence recognition or relation extraction
- run_ksr_re_train.py
- run_ksr_re_test.py
  
Python codes for named entity recognition
- run_ner_train.py
- run_ner_test.py


## Requirements
- Python 3.8 or higher
- PyTorch 2.0 or higher
- Transformers 4.3 or higher
- Pandas 1.2 or higher
- skikit-learn 1.4 or higher
- seqeval 1.2 or higher

## Citation
To be added
