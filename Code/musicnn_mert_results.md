# **MusiCNN & MERT Model Comparison Results**

***LOGISTIC REGRESSION***


**Logistic Regression Raw Summary**
| Model   | Val. Acc. | Val. Macro F1 | Test Acc. | Test Macro F1 | # Parameters |
|----------|----------:|--------------:|----------:|--------------:|-------------:|
| Musicnn | **85.82%** | **0.8605** | **84.31%** | **0.8453** | 7,530 |
| MERT     | 82.27% | 0.8189 | 76.47% | 0.7609 | 7,680 |

---

**Logistic Regression Performance Summary**
| Metric              | Better Model |              Difference |
| ------------------- | ------------ | ----------------------: |
| Validation Accuracy | Musicnn      | +3.55 percentage points |
| Validation Macro F1 | Musicnn      |                 +0.0416 |
| Test Accuracy       | Musicnn      | +7.84 percentage points |
| Test Macro F1       | Musicnn      |                 +0.0844 |
| Parameters          | Musicnn      |               150 fewer |

---

***SMALL MLP***

**Small MLP Raw Summary**
| Model   | Val. Acc. | Val. Macro F1 | Test Acc. | Test Macro F1 | # Parameters |
|----------|----------:|--------------:|----------:|--------------:|-------------:|
| Musicnn | **89.36%** | **0.8947** | **86.27%** | **0.8644** | 97,802 |
| MERT     | 84.40% | 0.8430 | 74.51% | 0.7407 | 99,722 |

---

**Small MLP Performance Summary**
| Metric              | Better Model |               Difference |
| ------------------- | ------------ | -----------------------: |
| Validation Accuracy | Musicnn      |  +4.96 percentage points |
| Validation Macro F1 | Musicnn      |                  +0.0517 |
| Test Accuracy       | Musicnn      | +11.76 percentage points |
| Test Macro F1       | Musicnn      |                  +0.1237 |
| Parameters          | Musicnn      |              1,920 fewer |

---

***LOGISTIC REGRESSION | SMALL MLP SIDE-BY-SIDE***

| Metric | Logistic Regression | MLP |
|--------|--------------------|--------------------|
| Better Model | Musicnn | Musicnn |
| Validation Accuracy | +3.55 pp | +4.96 pp |
| Validation Macro F1 | +0.0416 | +0.0517 |
| Test Accuracy | +7.84 pp | +11.76 pp |
| Test Macro F1 | +0.0844 | +0.1237 |
| Parameter Difference | 150 fewer | 1,920 fewer |

-- ***pp=*** percentage points

---

***CONCLUSION***

For both logistic regression and small MLP training, **musicnn** outperformed ***MERT***. The logistic regression results show musicnn doing better on both validation and test sets. Additionally, it uses slightly less model parameters. The MLP results show an even bigger gap, **musicnn achieved 86.27% test accuracy** compared to *74.51%* for MERT, while also using **1,920 fewer parameters**. **musicnn is the clear winner with a major performance advantage when using the MLP classifier**