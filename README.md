# Medical-Concept-Mapping

An implementation of **Medical-term Concept Mapping** via three-levels: Syntax-Semantics-Pragmatics.

------

## Method

Overall Method is shown below:

<p align="center">
  <img src='Method.png'>
</p>

Specific Method:

1. **Syntax-level**: Sub-word Frequency via BPE Algorithm 

2. **Semantics-level**: Word vector Cosine Similarity
    
    <p align="center">
      <img src='demo.png'>
    </p>

3. **Pragmatics-level**: Knowledge Graph

    <p align="center">
      <img src='Knowledge-Graph.png'>
    </p>

------

## Results

**96.81% Accuracy** on the Standard and Synonym Medical Terms

------

## Prerequisites

The pre-trained word vectors can be downloaded [here](https://drive.google.com/file/d/1b_D5OQHm1XFlHKcMaWUJ8ABiQNPM0meS/view?usp=sharing).

------

## Presentation

A part presentation of this work can be downloaded [here](https://github.com/SuperBruceJia/paper-reading/raw/master/NLP-field/Sub-words/Concept-Matching-Task.pptx).

------

## Acknowledgement

This work was done when I was in Philips Research Shanghai.
