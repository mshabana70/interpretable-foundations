# Interpretable Foundations: AI Safety & Security Study Roadmap

 A discipline-first roadmap to AI safety research competence spanning 42 weeks (~10.5 months).  This curriculum bridges foundational mathematics and low-level ML implementations with mechanistic interpretability, causal inference, and adversarial machine learning.

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

*  **Daily Allocation:** 1 hour of focused AI study + 1 hour separate CPTS block.
*  **Session Structure:** ~25 min Conceptual Input 📖 → ~30 min Implementation 💻 → ~5 min Reflection 📝.
*  **The Golden Rule:** Build, don’t memorize. 
*  **Requirement:** Every concept gets implemented in code. 
*  **Validation:** If you can’t rebuild it from scratch in NumPy or PyTorch, you don’t understand it yet.

---

## 🗺️ Detailed Phase Tracker

### 🟥 Phase 1: Mathematical Intuition & NumPy Fluency (Weeks 1–6)
 **Goal:** Develop geometric intuition for linear algebra and calculus, and become fluent in NumPy as your “math scratchpad.” 

####  Week 1: Vectors, Spaces & NumPy Foundations 
**Theme:** What are vectors, really?  And how do we compute with them? 

| Day | Conceptual | Implementation Artifact | Status |
| :--- | :--- | :--- | :---: |
| **Mon** | 3B1B Essence of LinAlg: Ep 1 & 2 |  `vectors.py` (Pure Python `Vector` class)  | [ ] |
| **Tue** | 3B1B LinAlg: Ep 3 |  `matrix.py` (Extend Vector class to matrices)  | [ ] |
| **Wed** | MML Ch 2, Sec 2.1–2.3 |  Vectorized migration of Vector class to NumPy arrays  | [ ] |
| **Thu** | 3B1B LinAlg: Ep 4 & 5 |  Matplotlib 2D transformation grid visualizer  | [ ] |
| **Fri** | **FROM SCRATCH CHALLENGE** |  Matrix multiplication for arbitrary matrices in pure Python  | [ ] |
| **Sat** | MML Ch 2, Sec 2.4–2.6 |  Linear independence row reduction algorithm  | [ ] |

####  Week 2: Determinants, Inverses & Solving Systems 
**Theme:** When can we reverse a transformation?  What does a determinant tell us? 

| Day | Conceptual | Implementation Artifact | Status |
| :--- | :--- | :--- | :---: |
| **Mon** | 3B1B LinAlg: Ep 6 & 7 |  Determinant & 2x2 matrix inversion from scratch  | [ ] |
| **Tue** | 3B1B LinAlg: Ep 8 & 9 |  Gaussian elimination solving $Ax = b$  | [ ] |
| **Wed** | MML Ch 2, Sec 2.7–2.9 |  Inverse of arbitrary NxN matrix via augmented identity  | [ ] |
| **Thu** | 3B1B LinAlg: Ep 10 & 11 |  Linear system solver profiling solution spaces  | [ ] |
| **Fri** | **FROM SCRATCH CHALLENGE** |  Random 4x4 matrix determinant, inverse, and solve $Ax=b$  | [ ] |
| **Sat** | MML Ch 3, Sec 3.1–3.3 |  Norms and Cosine similarity nearest-neighbor search  | [ ] |

---

## 📝 Daily Study Log Template

 Copy this template into your daily commit message or a running `study_journal.md` file.

| Metric | Detail |
| :--- | :--- |
| **Date** | [YYYY-MM-DD] |
| **Phase & Topic** | [e.g., Phase 1 - Inverses] |
| **AI Study Time** |  [X] mins  |
| **CPTS Block** |  [Y] mins  |
| **Energy Level** |  [1-5]  |
| **What Clicked** |  [Brief insight]  |
| **Fuzzy Concepts** |  [Areas to revisit]  |