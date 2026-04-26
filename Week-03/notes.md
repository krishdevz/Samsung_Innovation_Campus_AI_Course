# 📖 Week 03 Notes — Probability & Statistics
**Chapter:** 4 | **Course:** Samsung Innovation Campus AI

---

## Chapter 4: Probability & Statistics

> 🧠 **Why this matters:** ML models deal with uncertainty constantly. "How confident is the model?" "Is this pattern real or just random noise?" "How do we compare two groups?" — all of these are answered using probability and statistics.

---

## Part A: Probability

### 🔷 4.1 What is Probability?
Probability measures **how likely** an event is to happen. It is a number between 0 and 1.

```
P(event) = 0       → impossible (will never happen)
P(event) = 0.5     → 50-50 chance
P(event) = 1       → certain (will always happen)
```

> 🧠 **Analogy:** Flip a fair coin. P(Heads) = 0.5. Roll a dice. P(getting 6) = 1/6 ≈ 0.167.

---

### 🔷 4.2 Basic Probability Rules

**Rule 1 — Complement Rule:**
```
P(not A) = 1 − P(A)

Example: P(rain) = 0.3  →  P(no rain) = 1 − 0.3 = 0.7
```

**Rule 2 — Addition Rule (OR):**
```
P(A or B) = P(A) + P(B) − P(A and B)

If A and B are mutually exclusive (can't happen together):
P(A or B) = P(A) + P(B)
```

**Rule 3 — Multiplication Rule (AND):**
```
If A and B are independent (one doesn't affect the other):
P(A and B) = P(A) × P(B)

Example: P(Heads AND Heads on two flips) = 0.5 × 0.5 = 0.25
```

---

### 🔷 4.3 Conditional Probability
The probability of event A happening **given that** event B has already happened.

```
P(A | B) = P(A and B) / P(B)

Read as: "Probability of A given B"
```

> 📌 **Example:** In a class of 30 students, 12 are girls, 8 girls passed. What is the probability that a student passed, given they are a girl?
> P(Pass | Girl) = P(Pass AND Girl) / P(Girl) = (8/30) / (12/30) = 8/12 = 0.67

> 🧠 **Why it matters in ML:** Naive Bayes classifier is entirely built on conditional probability. Also used in Bayesian models and many evaluation metrics.

---

### 🔷 4.4 Bayes' Theorem
Updates the probability of a hypothesis based on new evidence.

```
P(A | B) = [ P(B | A) × P(A) ] / P(B)
```

| Term | Name | Meaning |
|------|------|---------|
| P(A) | Prior | Initial belief before seeing evidence |
| P(B\|A) | Likelihood | Probability of evidence if A is true |
| P(B) | Marginal | Total probability of evidence |
| P(A\|B) | Posterior | Updated belief after seeing evidence |

> 📌 **Example:** Medical test is 99% accurate. Disease affects 1% of population. If you test positive, what is the actual probability you have the disease? (Answer: surprisingly low — Bayes' theorem shows this!)

---

### 🔷 4.5 Types of Events

| Type | Definition | Example |
|------|-----------|---------|
| **Independent** | One event doesn't affect the other | Two coin flips |
| **Dependent** | One event affects probability of other | Drawing cards without replacement |
| **Mutually Exclusive** | Both can't happen at same time | Rolling 1 AND 6 on same dice throw |
| **Exhaustive** | Together they cover all possibilities | Heads OR Tails |

---

### 🔷 4.6 Random Variables
A variable whose value is determined by the outcome of a random event.

**Discrete Random Variable** — takes specific, countable values:
```
X = number of heads in 3 coin flips
X can be: 0, 1, 2, or 3
```

**Continuous Random Variable** — takes any value in a range:
```
X = height of a person
X can be: 170.0, 170.1, 170.15, ... (infinite possibilities)
```

---

### 🔷 4.7 Expected Value (Mean of Random Variable)
The average value you'd expect if you repeated the experiment many times.

```
E(X) = Σ [ x × P(x) ]

Example: Dice roll
E(X) = 1×(1/6) + 2×(1/6) + 3×(1/6) + 4×(1/6) + 5×(1/6) + 6×(1/6)
     = 21/6 = 3.5
```

---

## Part B: Probability Distributions

### 🔷 4.8 What is a Distribution?
A distribution describes **how values are spread** across all possible outcomes. It tells you which values are common and which are rare.

---

### 🔷 4.9 Normal (Gaussian) Distribution
The most important distribution in statistics and ML.

**Properties:**
- Bell-shaped, perfectly symmetric
- Mean = Median = Mode (all at the centre)
- 68% of data falls within 1 std dev of mean
- 95% within 2 std devs
- 99.7% within 3 std devs (the "68-95-99.7 rule")

```
The 68-95-99.7 Rule:
|←  68%  →|
|←    95%    →|
|←      99.7%      →|
  μ-3σ  μ-2σ  μ-σ  μ  μ+σ  μ+2σ  μ+3σ
```

> 📌 **Example:** Heights of adults, exam scores, measurement errors — all follow approximately normal distributions.

> 🧠 **Why it matters:** Many ML algorithms assume data is normally distributed. Checking this assumption helps you choose the right model.

---

### 🔷 4.10 Skewed Distributions

**Positively Skewed (Right skew):**
- Long tail on the RIGHT
- Mean > Median > Mode
- Example: Income distribution (most earn less, a few earn very high)

**Negatively Skewed (Left skew):**
- Long tail on the LEFT
- Mean < Median < Mode
- Example: Age at retirement (most retire around 60, a few retire very early)

> ⭐ **Remember:** If mean ≠ median, distribution is skewed. Use median for skewed data, not mean!

---

### 🔷 4.11 Binomial Distribution
Used when you have a fixed number of **independent trials**, each with two outcomes (success/failure).

```
Parameters:
  n = number of trials
  p = probability of success in each trial

Example: Flip coin 10 times. What's P(exactly 7 heads)?
  n=10, p=0.5
```

> 📌 **ML use:** Modelling binary classification outcomes, quality control.

---

### 🔷 4.12 Uniform Distribution
Every outcome is equally likely.

```
Example: Rolling a fair dice → P(1) = P(2) = ... = P(6) = 1/6
```

---

## Part C: Inferential Statistics

### 🔷 4.13 Population vs Sample

| Term | Definition | Example |
|------|-----------|---------|
| **Population** | The complete group you want to study | All students in India |
| **Sample** | A subset of the population | 500 students surveyed |
| **Parameter** | A number describing the population | True mean height of all Indians |
| **Statistic** | A number describing the sample | Mean height of our 500 students |

> 🧠 **Key insight:** In ML, your training data is a SAMPLE of the real world. The goal is to build a model that works on the full POPULATION (new, unseen data).

---

### 🔷 4.14 Hypothesis Testing
A method to decide whether your data supports a claim or hypothesis.

**Steps:**
```
1. Set up Null Hypothesis H₀  →  the "no effect / no difference" claim
2. Set up Alt Hypothesis H₁   →  what you're trying to prove
3. Choose significance level α  →  usually 0.05 (5%)
4. Calculate test statistic from your data
5. Calculate p-value
6. Decision: if p-value < α → reject H₀ (result is significant)
             if p-value ≥ α → fail to reject H₀
```

> 📌 **Example:**
> H₀: "New teaching method has no effect on scores"
> H₁: "New teaching method improves scores"
> If p-value = 0.02 < 0.05 → reject H₀ → teaching method works!

---

### 🔷 4.15 P-Value
The probability of getting your observed results (or more extreme) **if the null hypothesis is true**.

```
Small p-value (< 0.05)  →  result is statistically significant → reject H₀
Large p-value (≥ 0.05)  →  result could be due to chance → don't reject H₀
```

> ⚠️ **Important:** p-value < 0.05 does NOT mean the effect is large or practically important. It just means it's unlikely to be due to chance.

---

### 🔷 4.16 Confidence Intervals
A range of values that likely contains the true population parameter.

```
"95% confidence interval for mean score: [72, 85]"
→ We are 95% confident the true population mean lies between 72 and 85
```

> 🧠 **Analogy:** Instead of saying "the average height is exactly 170 cm", you say "I'm 95% sure the average is between 168 and 172 cm." More honest and realistic.

---

### 🔷 4.17 Correlation
Measures the **strength and direction** of the relationship between two variables. Value ranges from −1 to +1.

```
Pearson Correlation: r = Σ[(x - x̄)(y - ȳ)] / [n × σx × σy]

r = +1 → perfect positive correlation (both increase together)
r =  0 → no linear relationship
r = −1 → perfect negative correlation (one increases, other decreases)
```

| r value | Interpretation |
|---------|---------------|
| 0.8 to 1.0 | Strong positive |
| 0.5 to 0.8 | Moderate positive |
| 0.2 to 0.5 | Weak positive |
| −0.2 to 0.2 | No correlation |
| −0.5 to −0.2 | Weak negative |
| −0.8 to −0.5 | Moderate negative |
| −1.0 to −0.8 | Strong negative |

> ⚠️ **CRITICAL:** Correlation ≠ Causation! Ice cream sales and drowning deaths are correlated (both increase in summer) — but ice cream doesn't cause drowning!

---

### 🔷 4.18 Covariance
Measures whether two variables move together. Similar to correlation but not normalised.

```
Cov(X, Y) > 0  →  X and Y tend to increase together
Cov(X, Y) < 0  →  when X increases, Y tends to decrease
Cov(X, Y) = 0  →  no linear relationship
```

> ⭐ **Correlation vs Covariance:** Correlation is just covariance divided by (σx × σy) — it's a normalised version that's easier to interpret because it's always between −1 and +1.

---

## 💡 Key Takeaways — Week 3
1. Probability = likelihood of an event, always between 0 and 1
2. Conditional probability P(A|B) = probability of A given B occurred
3. Bayes' theorem updates beliefs based on new evidence — foundation of Naive Bayes ML
4. Normal distribution (bell curve) is the most important distribution — understand the 68-95-99.7 rule
5. Skewed data → use median, not mean
6. Hypothesis testing: p-value < 0.05 = reject null hypothesis = statistically significant
7. Correlation measures relationship strength (−1 to +1) — but correlation ≠ causation!
8. In ML: your dataset is a SAMPLE — your goal is to generalise to the POPULATION
