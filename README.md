# Interpretable Foundations: AI Safety & Security Study Roadmap

[cite_start]A discipline-first roadmap to AI safety research competence spanning 42 weeks (~10.5 months)[cite: 2]. [cite_start]This curriculum bridges foundational mathematics and low-level ML implementations with mechanistic interpretability, causal inference, and adversarial machine learning[cite: 125, 141].

## 📊 Progress Dashboard

![Study Streak](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/mshabana70/interpretable-foundations/main/streak.json)

| Phase | Focus Domain | Duration | Status |
| :---: | :--- | :---: | :---: |
| **1** | Math Foundations & NumPy Fluency | Weeks 1–6 | 🔄 In Progress |
| **2** | ML Algorithms from Scratch | Weeks 7–12 | 🛑 Unstarted |
| **3** | Deep Learning, Tokenization, Transformers & Systems | Weeks 13–19 | 🛑 Unstarted |
| **4** | Adversarial ML & Model Security | Weeks 20–23 | 🛑 Unstarted |
| **5** | ARENA, Experimental Methods & Causal Inference | Weeks 24–29 | 🛑 Unstarted |
| **6** | Information Theory, Representations & Training Dynamics | Weeks 30–34 | 🛑 Unstarted |
| **7** | RL, RLHF, LoRA & the Alignment Pipeline | Weeks 35–38 | 🛑 Unstarted |
| **8** | AI Safety Theory, Red-Teaming & Evals | Weeks 39–42 | 🛑 Unstarted |

---

## 🛠️ Execution Protocol

* [cite_start]**Daily Allocation:** 1 hour of focused AI study + 1 hour separate CPTS block[cite: 4].
* [cite_start]**Session Structure:** ~25 min Conceptual Input 📖 → ~30 min Implementation 💻 → ~5 min Reflection 📝[cite: 13].
* [cite_start]**The Golden Rule:** Build, don’t memorize[cite: 9]. 
* [cite_start]**Requirement:** Every concept gets implemented in code[cite: 9]. 
* [cite_start]**Validation:** If you can’t rebuild it from scratch in NumPy or PyTorch, you don’t understand it yet[cite: 9].

---

## 🗺️ Detailed Phase Tracker

### 🟥 Phase 1: Mathematical Intuition & NumPy Fluency (Weeks 1–6)
[cite_start]**Goal:** Develop geometric intuition for linear algebra and calculus, and become fluent in NumPy as your “math scratchpad.” [cite: 24]

#### [cite_start]Week 1: Vectors, Spaces & NumPy Foundations [cite: 33]
**Theme:** What are vectors, really? [cite_start]And how do we compute with them? [cite: 34]

| Day | Conceptual | Implementation Artifact | Status |
| :--- | :--- | :--- | :---: |
| **Mon** | 3B1B Essence of LinAlg: Ep 1 & 2 | [cite_start]`vectors.py` (Pure Python `Vector` class) [cite: 35] | [ ] |
| **Tue** | 3B1B LinAlg: Ep 3 | [cite_start]`matrix.py` (Extend Vector class to matrices) [cite: 35] | [ ] |
| **Wed** | MML Ch 2, Sec 2.1–2.3 | [cite_start]Vectorized migration of Vector class to NumPy arrays [cite: 35] | [ ] |
| **Thu** | 3B1B LinAlg: Ep 4 & 5 | [cite_start]Matplotlib 2D transformation grid visualizer [cite: 35] | [ ] |
| **Fri** | **FROM SCRATCH CHALLENGE** | [cite_start]Matrix multiplication for arbitrary matrices in pure Python [cite: 35] | [ ] |
| **Sat** | MML Ch 2, Sec 2.4–2.6 | [cite_start]Linear independence row reduction algorithm [cite: 35] | [ ] |

#### [cite_start]Week 2: Determinants, Inverses & Solving Systems [cite: 36]
**Theme:** When can we reverse a transformation? [cite_start]What does a determinant tell us? [cite: 37]

| Day | Conceptual | Implementation Artifact | Status |
| :--- | :--- | :--- | :---: |
| **Mon** | 3B1B LinAlg: Ep 6 & 7 | [cite_start]Determinant & 2x2 matrix inversion from scratch [cite: 38] | [ ] |
| **Tue** | 3B1B LinAlg: Ep 8 & 9 | [cite_start]Gaussian elimination solving $Ax = b$ [cite: 38] | [ ] |
| **Wed** | MML Ch 2, Sec 2.7–2.9 | [cite_start]Inverse of arbitrary NxN matrix via augmented identity [cite: 38] | [ ] |
| **Thu** | 3B1B LinAlg: Ep 10 & 11 | [cite_start]Linear system solver profiling solution spaces [cite: 38] | [ ] |
| **Fri** | **FROM SCRATCH CHALLENGE** | [cite_start]Random 4x4 matrix determinant, inverse, and solve $Ax=b$ [cite: 38] | [ ] |
| **Sat** | MML Ch 3, Sec 3.1–3.3 | [cite_start]Norms and Cosine similarity nearest-neighbor search [cite: 38] | [ ] |

---

## 📝 Daily Study Log Template

[cite_start]Copy this template into your daily commit message or a running `study_journal.md` file[cite: 19].

| Metric | Detail |
| :--- | :--- |
| **Date** | [YYYY-MM-DD] |
| **Phase & Topic** | [e.g., Phase 1 - Inverses] |
| **AI Study Time** | [cite_start][X] mins [cite: 309] |
| **CPTS Block** | [cite_start][Y] mins [cite: 309] |
| **Energy Level** | [cite_start][1-5] [cite: 309] |
| **What Clicked** | [cite_start][Brief insight] [cite: 18] |
| **Fuzzy Concepts** | [cite_start][Areas to revisit] [cite: 18] |