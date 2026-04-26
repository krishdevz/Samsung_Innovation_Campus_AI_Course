# 📖 Week 01 Notes — Introduction to AI & Math for Data Science
**Chapters:** 1 & 2 | **Course:** Samsung Innovation Campus AI

---

## Chapter 1: Introduction to Artificial Intelligence

### 🔷 1.1 What is AI?
Artificial Intelligence is the science of building machines that can perform tasks requiring **human-like intelligence** — reasoning, learning, understanding language, recognising images, and making decisions.

> 🧠 **Analogy:** Your brain recognises a cat in a photo instantly. AI teaches a computer to do the same — but using math and data instead of biology.

**What AI can do:**
- Understand and generate language → chatbots, Google Translate, autocorrect
- Recognise images and faces → Face ID, Google Photos
- Make recommendations → YouTube, Netflix, Amazon
- Detect diseases from medical scans
- Drive cars autonomously

---

### 🔷 1.2 AI vs ML vs Deep Learning vs Data Science

| Term | What it means | Example |
|------|--------------|---------|
| **AI** | Any technique making machines mimic human intelligence | Chess-playing program |
| **Machine Learning** | Machines learn from data without being explicitly programmed | Spam filter |
| **Deep Learning** | ML using multi-layered neural networks for complex patterns | Face recognition |
| **Data Science** | Extracting insights from data using stats, ML, visualisation | Sales trend analysis |

> ⭐ **Remember:** AI ⊇ Machine Learning ⊇ Deep Learning  
> Every DL model is an ML model. Every ML model is an AI system.

---

### 🔷 1.3 Three Types of Machine Learning

| Type | Input | Goal | Example |
|------|-------|------|---------|
| **Supervised** | Input + correct labels | Predict output for new inputs | Predict house price |
| **Unsupervised** | Input only, no labels | Discover hidden patterns | Customer segmentation |
| **Reinforcement** | Rewards & penalties | Learn strategy to maximise reward | Teaching robot to walk |

**Supervised Learning tasks:**
- **Classification** → predict a category (spam/not spam, cat/dog, disease/healthy)
- **Regression** → predict a number (house price, temperature, salary)

**Unsupervised Learning tasks:**
- **Clustering** → group similar data points (customer segments)
- **Dimensionality Reduction** → reduce features, keep key info (PCA)

---

### 🔷 1.4 The ML Pipeline
Every ML project follows these steps:
```
1. Collect Data
2. Clean & Preprocess
3. Select Features
4. Choose Model
5. Train
6. Evaluate
7. Predict / Deploy
```

---

### 🔷 1.5 Real-World Applications
- **Google Search** → understands intent, not just keywords
- **Gmail** → spam filter using Naive Bayes ML
- **Netflix/YouTube** → recommendation engines
- **Face ID** → deep learning for face recognition
- **Fraud Detection** → banks flag unusual transactions
- **Medical Imaging** → AI detects tumours in X-rays

---

## Chapter 2: Math for Data Science

### 🔷 2.1 Why Math Matters
All data — images, text, audio — gets converted into **numbers**. ML models compute on these numbers. Math is the language that lets you understand and build these computations.

- **Linear Algebra** → represent and transform data
- **Statistics** → summarise and understand data
- **Probability** → model uncertainty and confidence
- **Derivatives/Calculus** → how models actually learn

---

### 🔷 2.2 Data Structures

| Structure | Dimensions | Description | Example |
|-----------|-----------|-------------|---------|
| **Scalar** | 0D | Single number | Temperature: 36.6 |
| **Vector** | 1D | Array of numbers | [Age=21, Height=170, Score=88] |
| **Matrix** | 2D | Table (rows × cols) | Dataset: 100 students × 5 features |
| **Tensor** | 3D+ | Multi-dimensional | Image: 224×224×3 (RGB channels) |

> ⭐ **Remember:** In ML datasets — each **ROW** = one sample. Each **COLUMN** = one feature.

---

### 🔷 2.3 Dot Product
Multiplies two vectors element-by-element and sums the results. Returns a single number.

```
Formula:  a · b = (a1×b1) + (a2×b2) + ... + (an×bn)

Example:  a = [1, 2, 3]   b = [4, 5, 6]
          dot = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

> 🧠 **Analogy:** In ML, a prediction = dot product of feature values × their weights. That's literally what a linear model does!

---

### 🔷 2.4 Matrix Multiplication
Combines two matrices. Core operation in every neural network.

```
Rule: A (m×n) × B (n×p) = C (m×p)
      Inner dimensions MUST match!

C[i][j] = Σ A[i][k] × B[k][j]
```

> ⚠️ **Important:** A×B ≠ B×A (order matters in matrix multiplication!)

---

### 🔷 2.5 Transpose
Flip a matrix — rows become columns, columns become rows. Written as A^T.

```
A = [[1, 2, 3],        A^T = [[1, 4],
     [4, 5, 6]]               [2, 5],
                               [3, 6]]
```

---

### 🔷 2.6 Descriptive Statistics

| Measure | Formula | What it tells you |
|---------|---------|------------------|
| **Mean** | Σx / n | Centre of data. Sensitive to outliers |
| **Median** | Middle value when sorted | More robust centre when outliers exist |
| **Mode** | Most frequent value | Best for categorical data |
| **Range** | max − min | Total spread |
| **Variance (σ²)** | Σ(x − μ)² / n | Average squared distance from mean |
| **Std Deviation (σ)** | √Variance | Average distance from mean (same units as data) |

```
Example: Data = [10, 20, 30, 40, 50]
  Mean     = (10+20+30+40+50)/5 = 30
  Variance = [(10-30)²+(20-30)²+(30-30)²+(40-30)²+(50-30)²] / 5
           = [400+100+0+100+400] / 5 = 200
  Std Dev  = √200 = 14.14
```

> 🧠 **Analogy:** Std Dev tells you "on average, how far are values from the mean?" Small = tightly clustered. Large = widely spread.

---

### 🔷 2.7 Normalization vs Standardization
ML algorithms are sensitive to scale. Age (0–100) vs Salary (0–100,000) — salary would dominate unfairly.

**Min-Max Normalization** → scales to [0, 1]
```
X_scaled = (X − X_min) / (X_max − X_min)

Example: [10, 20, 30] → [0.0, 0.5, 1.0]
```
- ✅ Good for: Neural Networks, KNN
- ❌ Problem: Sensitive to outliers

**Z-Score Standardization** → centres at 0, std dev = 1
```
X_scaled = (X − mean) / std_deviation

Example: [10, 20, 30], mean=20, std=8.16 → [−1.22, 0.0, 1.22]
```
- ✅ Good for: Linear models, SVM, PCA
- ✅ More robust to outliers

| Feature | Min-Max | Z-Score |
|---------|---------|---------|
| Output range | [0, 1] | No fixed range |
| Outlier sensitivity | High | Low |
| Best for | KNN, Neural Nets | Linear models, SVM |

---

### 🔷 2.8 Derivatives & Gradient Descent ⭐ (Very Important!)

#### What is a Derivative?
A derivative measures **rate of change** — how much the output changes when input changes slightly. It gives the **slope** of a function at any point.

```
If y = x²  →  derivative dy/dx = 2x

At x=3:  slope = 6   (rising steeply)
At x=0:  slope = 0   (flat — this is the minimum!)
```

#### Loss Function
Measures how WRONG the model's predictions are. The model wants to **minimise** this.

```
Example: Predicted = 50,  Actual = 60
         Error = 10,  Loss = Error² = 100
```

#### Gradient Descent — How Models Learn
Uses derivatives to find weights that minimise the loss:

```
Step 1: Start with random weights
Step 2: Make predictions
Step 3: Calculate loss
Step 4: Compute gradient (derivative of loss w.r.t. weights)
Step 5: Update weights → New Weight = Old Weight − (Learning Rate × Gradient)
Step 6: Repeat until loss stops decreasing
```

> 🧠 **Hiking analogy:** You're blindfolded on a mountain. You feel the slope (gradient), take one step downhill (weight update), repeat until you reach the valley (minimum loss).

#### Learning Rate
- Too **large** → overshoot minimum, model never converges
- Too **small** → learning is extremely slow
- Just **right** → model converges efficiently

#### Types of Gradient Descent

| Type | How it works | Best for |
|------|-------------|----------|
| Batch GD | Uses ALL data at once | Small datasets |
| Stochastic GD (SGD) | One sample at a time | Large datasets |
| Mini-Batch GD | Small batches (e.g. 32) | Most practical — used everywhere |

#### Chain Rule
For computing gradients through multiple layers in deep networks:
```
If z = f(g(x))  →  dz/dx = f'(g(x)) × g'(x)
```
> ⭐ This is the foundation of **Backpropagation** — how deep neural networks learn!

---

## 💡 Key Takeaways — Week 1
1. AI ⊇ ML ⊇ Deep Learning — understand this hierarchy
2. Three ML types: Supervised (labels), Unsupervised (no labels), Reinforcement (rewards)
3. All data is represented as vectors, matrices, or tensors
4. Std Dev measures spread; Mean measures centre
5. Gradient Descent is HOW models learn — minimise loss by moving downhill
6. Learning Rate controls speed of learning — too big or too small is bad
7. Always scale your data (Normalise or Standardise) before training
