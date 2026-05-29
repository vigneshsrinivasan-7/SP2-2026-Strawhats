# 🏏 IPL Powerplay Score Predictor

A machine learning model that predicts the **total runs scored in the powerplay (first 6 overs)**
of an IPL match, given the batting team, bowling team, venue, and innings number.

---

## 📌 Problem Statement

Predict the powerplay score for an IPL match before it starts, using only:
- Batting team name
- Bowling team name
- Venue / stadium
- Innings number (1 or 2)

**Output:** A predicted integer score in the range **[55, 65]**

---

## 🧠 Approach

Instead of a pure black-box ML model, we use a **structured lookup-based scoring system**
built on historical IPL data and live IPL 2026 match data.

### Three Core Signals

| Signal | Description | Weight |
|---|---|---|
| **Batting Strength** | How many runs this team scores in PP on average | 50% |
| **Bowling Defence** | How many runs this bowling team concedes in PP | 30% |
| **Venue Factor** | Historical and 2026 PP average at this ground | 20% |

### Prediction Formula
