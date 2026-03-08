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
| 2021 | 128 | 68 | 60 | ⚠️ **53.1%** |
| 2022 | 138 | 72 | 66 | ⚠️ **52.2%** |
| 2023 | 137 | 91 | 46 | ✅ **66.4%** |
| 2024 | 84 | 35 | 49 | ❌ **41.7%** |
| 2025 | 112 | 60 | 52 | ⚠️ **53.6%** |
| **5-Year Total** | **599** | **326** | **273** | **54.4%** |

**2025 Verdict Breakdown:**
| Verdict | Signals | Correct | Accuracy |
|:---|:---:|:---:|:---:|
| UP 📈 | 99 | 54 | 54.5% |
| DOWN 📉 | 13 | 6 | 46.2% |
| SIDEWAYS ↔️ | 0 | 0 | 0.0% |

---
## 📈 Greedy (Conf > 40%)
| Year | Signals | Correct | Wrong | Accuracy |
|:---|:---:|:---:|:---:|:---:|
| 2021 | 229 | 127 | 102 | ✅ **55.5%** |
| 2022 | 238 | 119 | 119 | ⚠️ **50.0%** |
| 2023 | 232 | 137 | 95 | ✅ **59.1%** |
| 2024 | 214 | 105 | 109 | ❌ **49.1%** |
| 2025 | 190 | 90 | 100 | ❌ **47.4%** |
| **5-Year Total** | **1103** | **578** | **525** | **52.4%** |

**2025 Verdict Breakdown:**
| Verdict | Signals | Correct | Accuracy |
|:---|:---:|:---:|:---:|
| UP 📈 | 146 | 70 | 47.9% |
| DOWN 📉 | 44 | 20 | 45.5% |
| SIDEWAYS ↔️ | 0 | 0 | 0.0% |

---
## 📈 Gambler (Conf > 40%, No Filter)
| Year | Signals | Correct | Wrong | Accuracy |
|:---|:---:|:---:|:---:|:---:|
| 2021 | 244 | 137 | 107 | ✅ **56.1%** |
| 2022 | 243 | 121 | 122 | ❌ **49.8%** |
| 2023 | 237 | 139 | 98 | ✅ **58.6%** |
| 2024 | 236 | 117 | 119 | ❌ **49.6%** |
| 2025 | 245 | 116 | 129 | ❌ **47.3%** |
| **5-Year Total** | **1205** | **630** | **575** | **52.3%** |

**2025 Verdict Breakdown:**
| Verdict | Signals | Correct | Accuracy |
|:---|:---:|:---:|:---:|
| UP 📈 | 162 | 81 | 50.0% |
| DOWN 📉 | 83 | 35 | 42.2% |
| SIDEWAYS ↔️ | 0 | 0 | 0.0% |