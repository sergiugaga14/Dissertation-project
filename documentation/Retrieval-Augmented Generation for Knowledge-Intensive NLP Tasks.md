# Dissertation Thesis Notes

## Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

### Introduction

The paper *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* introduces Retrieval-Augmented Generation (RAG), a framework that combines the strengths of large language models (LLMs) with external knowledge retrieval systems.

The authors argue that large language models store knowledge in their learned parameters (referred to as **parametric memory**). While this allows them to encode a large amount of information, several limitations exist:

* Knowledge stored in model parameters is difficult to update.
* Models are prone to generating hallucinations.
* They provide limited interpretability regarding the source of their predictions.

To address these issues, the paper highlights **hybrid models** that combine parametric memory with **non-parametric (retrieval-based) memory**. Such systems allow knowledge to be:

* Updated without retraining the entire model.
* Expanded with new information.
* Inspected and interpreted through retrieved documents.

Previous hybrid approaches, such as **REALM** and **ORQA**, demonstrated promising results in knowledge-intensive NLP tasks.

The authors propose **Retrieval-Augmented Generation (RAG)**, a general-purpose fine-tuning framework that augments pre-trained sequence-to-sequence models with a non-parametric memory component.

In the RAG architecture:

* **Parametric memory** is provided by a pre-trained sequence-to-sequence Transformer model.
* **Non-parametric memory** consists of a dense vector index built from Wikipedia.
* Knowledge retrieval is performed using a pre-trained neural retriever.

---

## Methods

The official implementation of RAG is available in the Hugging Face Transformers repository:

https://github.com/huggingface/transformers/blob/master/examples/rag/

### Architecture

RAG consists of two main components:

#### 1. Retriever

The retriever identifies the top-*k* documents relevant to a query.

[
p(z|x)
]

Where:

* **x** = input query or prompt
* **z** = retrieved document(s)

The retriever returns the top-*k* most relevant documents for a given query.

#### 2. Generator

The generator produces the output sequence conditioned on both the query and the retrieved documents.

[
p(y_i \mid x, z, y_{1:i-1})
]

Where:

* **yᵢ** = current token being generated
* **y₁:ᵢ₋₁** = previously generated tokens
* **x** = input query
* **z** = retrieved document

The retriever and generator are trained jointly (**end-to-end**), with retrieved documents treated as latent variables.

---

## RAG Variants

The paper proposes two variants of RAG:

### RAG-Sequence

* Uses the same retrieved document(s) throughout the generation process.
* All output tokens are generated based on the same retrieved context.

### RAG-Token

* Allows different documents to be used for generating different tokens.
* Potentially provides more flexibility but increases computational complexity.

---

## Retriever

The retriever is based on **Dense Passage Retrieval (DPR)**, which uses a bi-encoder architecture:

* **BERTq**: query encoder
* **BERTd**: document encoder

The retriever performs:

### Maximum Inner Product Search (MIPS)

MIPS is used to identify the top-*k* documents whose vector representations have the highest similarity to the query vector.

---

## Generator

The generator is based on **BART**, a Transformer encoder-decoder architecture.

Input to the generator consists of the concatenation of:

* Query (**x**)
* Retrieved document (**z**)

The combined input is then used to generate the output sequence.

---

## Training

During training:

* The **BERTd** (document encoder) remains fixed.
* The **BERTq** (query encoder) is fine-tuned.
* The **BART** generator is fine-tuned.

This setup reduces computational cost while allowing the retriever and generator to adapt jointly.

---

## Decoding

The paper describes a decoding procedure for combining probabilities from retrieved documents during generation.

**Note:** The decoding section requires further study, as the methodology was not fully understood during the first reading.

---

## Experiments

### Knowledge Source

The non-parametric memory consists of a Wikipedia dump from 2018.

Key statistics:

* Approximately **21 million passages**
* Wikipedia articles split into chunks of roughly **100 words**
* Top-*k* documents retrieved during training (*k* = 5 or 10)

---

### Open-Domain Question Answering

The model is evaluated on several question-answering datasets:

* **Natural Questions (NQ)**
* **TriviaQA (TQA)**
* **WebQuestions (WQ)**
* **CuratedTREC (CT)**

Task format:

* Input: question
* Output: answer

---

### Abstractive Question Answering

The paper also evaluates RAG on abstractive question-answering tasks.

Dataset:

* **MS MARCO NLG v2.1**

This task focuses on generating natural language answers rather than extracting text spans directly from retrieved documents.

**Note:** Further reading is required to fully understand this evaluation setting.

---

### Jeopardy Question Answering

The paper evaluates RAG on Jeopardy-style questions.

These appear to be another form of open-domain question answering where answers must be generated from broad world knowledge.

---

### Fact Verification

The model is tested on fact verification tasks.

Objective:

* Determine whether a claim is:

  * **Supported**
  * **Refuted**

based on evidence retrieved from Wikipedia.

---

## Results

### Open-domain Question Answering

-> state of the art results on all the 
-> results in  table1
-> it combines the general flexibility of closes-book approaches (parametric) with the performance of open-book retrieval-based approaches

### Abstractive Question Answering

-> results in table2, for MS-MARCO 
-> it hallucinates less and the factually correct text generated is generated more often than in the case of BART
-> what are gold passages?

### Jeopardy Question Generation
-> I think Jeopary is a dataset, see results in table 2

### Fact Verification
-> results in table 2, FEVER datasets

### Additional Results
-> generation diversity 
-> retrieval ablation (experiment where you remove or alter a component of a system and see how that component affected the results/performance)
-> Index hot-swapping: knowledge can be easily changed at test time, compared to the parametric approach
-> Effect of Retrieving more documents : 5 or 10 documents, more, does not imply better results

## Related work

-> Single Task Retrieval: in prior work it was shown that retrieval can help for individual tasks, but in this case this it shows that it can do it across several tasks
-> learned retrieval 
-> general purpose architectures for NLP
-> memory-based architecture

## Conlusion

RAG - is a hybrid model that combines both parametric and non-parametric memory, and it is preffered by the people because it produceds more factully correct statements
