# 🔍 David Oracle: AI Signal Accuracy Audit

This audit answers ONE question: **When David says UP, DOWN, or SIDEWAYS, is he RIGHT?**

**Method**: For each trading day, David makes a prediction. We check the actual Nifty close 5 trading days later.
- **UP correct** = Nifty closed higher after 5 days
- **DOWN correct** = Nifty closed lower after 5 days
- **SIDEWAYS correct** = Nifty stayed within ±1% after 5 days

---
## 📈 Golden (Conf > 60%)
| Year | Signals | Correct | Wrong | Accuracy |
|:---|:---:|:---:|:---:|:---:|
| 2021 | 125 | 68 | 57 | ⚠️ **54.4%** |
| 2022 | 128 | 68 | 60 | ⚠️ **53.1%** |
| 2023 | 139 | 96 | 43 | ✅ **69.1%** |
| 2024 | 87 | 35 | 52 | ❌ **40.2%** |
| 2025 | 108 | 56 | 52 | ⚠️ **51.9%** |
| **5-Year Total** | **587** | **323** | **264** | **55.0%** |

**2025 Verdict Breakdown:**
| Verdict | Signals | Correct | Accuracy |
|:---|:---:|:---:|:---:|
| UP 📈 | 94 | 50 | 53.2% |
| DOWN 📉 | 14 | 6 | 42.9% |
| SIDEWAYS ↔️ | 0 | 0 | 0.0% |

---
## 📈 Greedy (Conf > 40%)
| Year | Signals | Correct | Wrong | Accuracy |
|:---|:---:|:---:|:---:|:---:|
| 2021 | 228 | 122 | 106 | ⚠️ **53.5%** |
| 2022 | 238 | 118 | 120 | ❌ **49.6%** |
| 2023 | 231 | 136 | 95 | ✅ **58.9%** |
| 2024 | 214 | 100 | 114 | ❌ **46.7%** |
| 2025 | 190 | 94 | 96 | ❌ **49.5%** |
| **5-Year Total** | **1101** | **570** | **531** | **51.8%** |

**2025 Verdict Breakdown:**
| Verdict | Signals | Correct | Accuracy |
|:---|:---:|:---:|:---:|
| UP 📈 | 144 | 71 | 49.3% |
| DOWN 📉 | 46 | 23 | 50.0% |
| SIDEWAYS ↔️ | 0 | 0 | 0.0% |

---
## 📈 Gambler (Conf > 40%, No Filter)
| Year | Signals | Correct | Wrong | Accuracy |
|:---|:---:|:---:|:---:|:---:|
| 2021 | 243 | 132 | 111 | ⚠️ **54.3%** |
| 2022 | 243 | 120 | 123 | ❌ **49.4%** |
| 2023 | 236 | 138 | 98 | ✅ **58.5%** |
| 2024 | 236 | 111 | 125 | ❌ **47.0%** |
| 2025 | 245 | 119 | 126 | ❌ **48.6%** |
| **5-Year Total** | **1203** | **620** | **583** | **51.5%** |

**2025 Verdict Breakdown:**
| Verdict | Signals | Correct | Accuracy |
|:---|:---:|:---:|:---:|
| UP 📈 | 155 | 79 | 51.0% |
| DOWN 📉 | 90 | 40 | 44.4% |
| SIDEWAYS ↔️ | 0 | 0 | 0.0% |