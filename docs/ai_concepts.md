# AI Concepts Guide for Freshers

A concise, interview-friendly overview of core Generative AI, Transformer, and RAG concepts designed for campus graduates and entry-level GenAI engineers.

---

## 1. Natural Language Processing (NLP)

### What is NLP?
Natural Language Processing (NLP) is a branch of artificial intelligence that enables computers to understand, interpret, generate, and manipulate human language.

### Core Stages in Text Processing
1. **Extraction**: Reading raw text from formats like PDF, DOCX, TXT, or HTML.
2. **Text Cleaning**: Stripping control characters, normalizing whitespace, and removing unwanted HTML/markup tags.
3. **Normalization**: Converting text to lower/standard case, removing accents, or standardizing punctuation.
4. **Tokenization**: Breaking clean text into discrete tokens (words, subwords, or characters).
5. **Embedding**: Converting tokens into numerical vectors for ML models.

---

## 2. Tokenization

### What is Tokenization?
Tokenization is the process of breaking raw text into smaller units called **tokens**. A token can be a word, a subword, or even a single character.

### Why not just use words?
- **Vocabulary Size**: Storing every unique word in English requires huge dictionaries.
- **Out-of-Vocabulary (OOV) Words**: Subword tokenization (e.g., Byte-Pair Encoding / BPE used by OpenAI's `tiktoken`) breaks unseen or compound words into recognized sub-tokens (`"unbelievable"` $\rightarrow$ `["un", "believ", "able"]`).

### Tokenization Flow
$$\text{Raw Text} \xrightarrow{\text{Tokenizer}} \text{Token Strings} \xrightarrow{\text{Vocabulary Lookup}} \text{Token IDs (Integers)}$$

Example using `tiktoken`:
- Input string: `"Generative AI Assistant"`
- Token IDs: `[38186, 2197, 15592, 17822]`

---

## 3. Embeddings & Vector Representations

### What is an Embedding?
An embedding is a dense mathematical vector (a list of floating-point numbers, e.g., dimension 1536) that captures the **semantic meaning** of text in a multi-dimensional concept space.

### Key Property: Semantic Proximity
In embedding space, words or chunks with similar meanings are positioned close together:
$$\text{Vector}(\text{"Resume"}) \approx \text{Vector}(\text{"CV"})$$
$$\text{CosineSimilarity}(\text{Vector}(\text{"Python"}), \text{Vector}(\text{"FastAPI"})) > \text{CosineSimilarity}(\text{Vector}(\text{"Python"}), \text{Vector}(\text{"Pizza"}))$$

### Cosine Similarity Formula
$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

---

## 4. Transformers & Self-Attention

### What is a Transformer?
Introduced in the seminal 2017 paper *"Attention Is All You Need"*, the Transformer architecture replaced traditional sequential models (RNNs/LSTMs) with a parallelizable, attention-based architecture.

### Self-Attention Mechanism
Self-attention calculates how strongly each word in a sequence relates to every other word in that same sequence.

For every input token vector, the model computes three vectors:
- **Query ($Q$)**: What am I looking for?
- **Key ($K$)**: What information do I hold?
- **Value ($V$)**: What representation do I pass forward?

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Where $d_k$ is the dimension of the key vectors (used for scaling to prevent vanishing gradients).

---

## 5. Context Window & LLMs

### What is an LLM?
A Large Language Model (LLM) is an autoregressive Transformer decoder trained on vast text corpora to predict the next token given a sequence of preceding tokens:
$$P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

### What is a Context Window?
The context window is the maximum number of tokens an LLM can process in a single request (prompt + completion).
- GPT-3.5-Turbo: 4,096 to 16,384 tokens
- GPT-4o: 128,000 tokens

If a document exceeds the context window, it cannot be fed directly into the prompt $\rightarrow$ **This is why chunking and RAG are required!**

---

## 6. Prompt Engineering

### What is Prompt Engineering?
Prompt Engineering is the practice of structuring text inputs to guide LLMs toward generating accurate, well-formatted, and relevant answers.

### Key Components of a Production Prompt
1. **System Role / Instruction**: Establishes persona, rules, and constraints (e.g., "Do not invent facts").
2. **User Profile**: Contextual personalization parameters.
3. **Retrieved Context**: Dynamic external data (e.g., top-K FAISS document chunks).
4. **User Question**: The primary query to be answered.

---

## 7. Retrieval-Augmented Generation (RAG)

### What is RAG?
Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances an LLM's capabilities by dynamically retrieving relevant facts from an external knowledge base (vector database) and embedding them into the prompt before generation.

### Why use RAG over Fine-Tuning?
| Feature | RAG | Fine-Tuning |
|---|---|---|
| **Data Freshness** | Instant (just update vector index) | Requires costly re-training |
| **Hallucination Control** | High (verifiable source citations) | Moderate to Low |
| **Cost** | Low (uses pre-trained API) | High (GPU compute required) |
| **Data Privacy** | Sensitive docs stay local in FAISS | Model weights store data permanently |

---

## 8. Vector Databases & FAISS

### What is FAISS?
FAISS (Facebook AI Similarity Search) is an open-source library built by Meta for efficient vector similarity search and clustering of dense vectors.

### How FAISS Works
1. Converts dense vectors into an indexed structure (`IndexFlatIP` for inner product / cosine distance).
2. Given a query embedding vector, FAISS calculates vector similarity against millions of stored embeddings in milliseconds.
3. Returns top-$K$ indices and similarity scores.
