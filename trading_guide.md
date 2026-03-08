# 📊 David Oracle — Options Trading Guide

> For **Bull Spreads, Bear Spreads, and Short Iron Condor** traders.
> Based on David v2.0 Hybrid Architecture (Regime Trees + LSTM).

---

## 🧠 How to Read David's Dashboard

David gives you 3 key signals. Here's what they mean and how to act:

### Signal 1: The Verdict (Direction)

| Verdict | What It Means | Your Action |
|:---|:---|:---|
| 🟢 **UP** | AI expects Nifty to close higher tomorrow | Bull Call Spread |
| 🔴 **DOWN** | AI expects Nifty to close lower tomorrow | Bear Put Spread |
| 🟡 **SIDEWAYS** | AI can't decide — market is flat | Short Iron Condor |

### Signal 2: Confidence (How Sure)

| Confidence | Conviction | Position |
|:---|:---|:---|
| **60%+** | ⭐ High conviction | Full 2-lot position |
| **50-60%** | ◆ Moderate | 1-lot position |
| **40-50%** | ○ Low conviction | Skip OR Iron Condor only |
| **< 40%** | ❌ No signal | **DO NOT TRADE** |

> **Important**: Today's screenshot shows **37% confidence** — this means David is essentially saying "I don't know." At this level, the only trade is a **Short Iron Condor** (betting on NO move).

### Signal 3: Regime

| Regime | What It Means | Best Strategy |
|:---|:---|:---|
| 🟢 **TRENDING** | Strong directional move underway | Follow the Verdict with directional spreads |
| 🟡 **CHOPPY** | Market grinding sideways | Iron Condors, Strangles — sell premium |
| 🔴 **VOLATILE** | Big swings both ways | Wider spreads, smaller size, OR skip |

### Why Dashboard Says SIDEWAYS But Forecast Shows UP

This is **not a bug** — they come from two completely different models:
- **Verdict**: Comes from the Hybrid AI (XGBoost + LSTM averaged). It classifies direction.
- **Price Forecast**: Comes from the Range Predictor (Quantile Regression). It projects price.

When confidence is low (37%), the Verdict says "SIDEWAYS" (uncertain), but the Range Predictor's median path still tilts slightly upward because historically, MILD BULLISH regimes drift up.

**How to interpret**: When they disagree at low confidence → **Short Iron Condor** (bet on range-bound).

---

## 📋 Daily Trading Checklist

```
□ Step 1: Open David Dashboard
□ Step 2: Check Confidence
    → If < 40%: SKIP (or Iron Condor only)
    → If 40-60%: Proceed with caution (1 lot)
    → If 60%+: Full conviction (2 lots)
□ Step 3: Check Regime
    → TRENDING: Directional spreads (Bull/Bear)
    → CHOPPY: Iron Condor
    → VOLATILE: Reduce size or skip
□ Step 4: Check Whipsaw
    → If CHOPPY: Don't chase breakouts
    → If TRENDING: Follow the signal
□ Step 5: Check Support/Resistance
    → Use these as strike selection anchors
□ Step 6: Enter trade (see strategy matrix below)
□ Step 7: Set exit rules (see below)
```

---

## 🎯 Strategy Matrix

### When Verdict = UP (Bull Call Spread)

| Component | Details |
|:---|:---|
| **Buy** | ATM or 1-strike ITM Call |
| **Sell** | 1-2 strikes OTM Call |
| **Spread Width** | 100-150 points |
| **Max Risk** | Spread width minus premium received |
| **Target** | 60-70% of max profit |
| **Stop** | Exit if Nifty breaks below nearest Support |
| **Holding** | 1-2 days (next-day prediction) |

**Example** (Nifty at 24,450):
- Buy 24400 CE
- Sell 24550 CE
- Net debit: ~₹55
- Max profit: ₹95 (if Nifty closes above 24550)
- Risk/Reward: 1:1.7

### When Verdict = DOWN (Bear Put Spread)

| Component | Details |
|:---|:---|
| **Buy** | ATM or 1-strike ITM Put |
| **Sell** | 1-2 strikes OTM Put |
| **Spread Width** | 100-150 points |
| **Target** | 60-70% of max profit |
| **Stop** | Exit if Nifty breaks above nearest Resistance |
| **Holding** | 1-2 days |

**Example** (Nifty at 24,450):
- Buy 24500 PE
- Sell 24350 PE
- Net debit: ~₹55
- Max profit: ₹95 (if Nifty closes below 24350)

### When Verdict = SIDEWAYS or Low Confidence (Short Iron Condor)

| Component | Details |
|:---|:---|
| **Sell** | OTM Call at nearest Resistance |
| **Buy** | OTM Call 100pts above (protection) |
| **Sell** | OTM Put at nearest Support |
| **Buy** | OTM Put 100pts below (protection) |
| **Premium** | ₹40-60 collected per lot |
| **Target** | 50% of premium (book at ₹20-30) |
| **Stop** | Exit if either short strike is breached |
| **Holding** | 1-3 days |

**Example** (Nifty at 24,450, Support 24,200, Resistance 24,850):
- Sell 24800 CE / Buy 24900 CE
- Sell 24250 PE / Buy 24150 PE
- Premium collected: ~₹50
- Max risk: ₹50 per side
- Target: Book at ₹25 premium remaining

---

## ⏱️ Expected Holding Period

Based on backtesting David V2 across 2025-2026:

| Strategy | Avg Holding | When To Exit |
|:---|:---|:---|
| **Bull/Bear Spread** | **1-2 days** | At 60-70% of max profit, OR next signal change |
| **Short Iron Condor** | **2-3 days** | At 50% premium decay, OR if strike breached |
| **Skip days** (low conf) | — | Don't trade below 40% confidence |

### Exit Rules:

1. **Profit Target Hit** → Exit immediately. Don't wait for more.
2. **Confidence Drops Sharply Next Day** (e.g., from 65% UP to 42% SIDEWAYS) → Close directional trade.
3. **Regime Changes** (e.g., TRENDING → VOLATILE) → Close all directional. Switch to Iron Condor.
4. **Never hold over expiry** if your strikes are anywhere near spot.
5. **"Will wait till profit"** works only on Iron Condors where time decay is your friend. On directional spreads, cut losses at 100% of premium paid.

---

## 📊 Capital Allocation Guide

| Capital | Per Trade | Max Open Positions |
|:---|:---|:---|
| ₹1 Lakh | 1 lot (₹5,000-7,000 margin) | 2 positions |
| ₹3 Lakh | 2 lots | 3 positions |
| ₹5 Lakh | 3-4 lots | 4 positions |
| ₹10 Lakh+ | 5 lots max | 5 positions |

**Rule**: Never risk more than 2% of capital on a single trade.

---

## ⚠️ When NOT to Trade

| Scenario | Why | What To Do |
|:---|:---|:---|
| Confidence < 40% | David doesn't know | Sit on hands |
| Budget Day / Union Budget | Unpredictable gap moves | Skip entirely |
| RBI Policy Day | Rate decisions cause spikes | Skip or Iron Condor only |
| F&O Expiry Day (Thursday) | Gamma risk, Pin risk | Avoid directional, close open trades |
| VIX > 25 | Market is wild | Half position size only |

---

## 🔑 Key Takeaways

1. **David predicts 1-day moves** — don't hold spreads for weeks.
2. **Confidence below 40% = No Trade**. The SIDEWAYS verdict at 37% means "skip."
3. **Iron Condor is your default** when David is uncertain. You profit from time passing.
4. **Always check Regime** — TRENDING markets deserve directional bets, CHOPPY markets deserve Iron Condors.
5. **Support/Resistance levels = your strike selection guide**. Sell options AT these levels.
6. **Average holding is 1-2 days**. This is not a swing trading system.

---

> [!IMPORTANT]
> David's accuracy is 62-66%. This means roughly 1 in 3 signals will be wrong. The key to profitability is **position sizing** (small losses) and **conviction filtering** (only trade high-confidence signals). With proper sizing, even 60% accuracy generates consistent returns over 100+ trades.

> [!CAUTION]
> **Paper trade for at least 2 weeks** before using real money. Track your signals against David's predictions to build confidence in the system.
