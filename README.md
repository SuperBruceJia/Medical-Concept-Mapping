# Medical Concept Mapping

An implementation of **Medical Concept Mapping** via three levels: **Syntax**-**Semantics**-**Pragmatics**.

## Abstract

Towards building an Artificial Intelligence-oriented (AI) healthcare system, precise mapping of medical concepts is highly demanded. Traditional works decoded medical terms lacking the consideration of a comprehensive overview of Natural Language Processing (NLP). However, for downstream NLP tasks, an analysis from different perspectives grows popular. In this work, a novel approach of medical concept mapping was presented from three aspects of NLP analysis, i.e., syntax, semantics, and pragmatics levels. Via the Byte Pair Encoding (BPE) Algorithm, the subwords' representations were introduced to learn the compounding and transliteration of medical concepts. Then, knowledge graph took advantages of human common sense in the perspective of pragmatics analysis. The final pre-trained word embedding and cosine similarity were utilized to map the input to the standard term which retain the maximum similarity. From the above three levels, the proposed approach has achieved compelling performance in the Chinese medical dataset, 96.81% accuracy. It indicated that our proposed method was able to handle the challenge of medical concept mapping, which can indirectly promoted the performance of healthcare AI systems.

## Method

Overall Method is shown below:

<p align="center">
  <img src='Method.png'>
</p>

Specific Method:

1. **Syntax-level**: Sub-word Frequency via BPE Algorithm 

2. **Semantics-level**: Word vector Cosine Similarity
    
    <p align="center">
      <img src='Demo.png'>
    </p>

3. **Pragmatics-level**: Knowledge Graph (JSON Format)

    <p align="center">
      <img src='Knowledge-Graph.png'>
    </p>

## Usage Demo

1. Get Sub-word (Frequency) list

```text
$ STEP-1-get-subword.py
```

2. Get Standard and Synonym Medical Terms

```text
$ STEP-2-get-Knowledge-Graph.py
```

3. Run the Concept Mapping main Function

```text
$ main.py
```

4. To evaluate, run the Evaluation Function

```text
$ evaluate.py
```

## Results

**96.81% Accuracy** on the Standard and Synonym Medical Terms

## Prerequisites

The pre-trained word vectors can be downloaded [here](https://drive.google.com/file/d/1Rfe7QObJnaOUYK3cIQ6BXLyuJg-5516Y/view?usp=sharing).

The data used for generating the sub-word list can be downloaded [here](https://drive.google.com/drive/folders/189J09QtpoAwrM9YUw1KGMaZSE6g_ukOc?usp=sharing).

## Presentation

A presentation of this work can be downloaded [here](https://github.com/SuperBruceJia/paper-reading/raw/master/NLP-field/Sub-words/Concept-Matching-Task.pptx).

## Acknowledgement

This work was done when I was in **Philips Research Shanghai**.
