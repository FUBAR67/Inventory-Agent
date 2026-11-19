# Inventory Decision Agent

An interactive decision-support application built in Streamlit for supply chain analytics, continuous and periodic inventory models, and multicriteria classification.

## Capabilities Included
1. **Classic EOQ**: Optimal lot sizing, cost curves, ordering vs. holding balance, deterministic ROP.
2. **All-Units Quantity Discounts**: Evaluates tiered supplier pricing, checks lot feasibility, and calculates minimum annual cost.
3. **Safety Stock & Reorder Point**: Continuous review $(ROP, Q)$ with stochastic demand and optional stochastic lead time.
4. **Fixed-Period Review Model**: Periodic review $(T, M)$ over protection interval $T + L$.
5. **Newsvendor Model**: Single-period stochastic inventory with critical fractile, expected excess/lost sales, and expected profit.
6. **Multicriteria Classification**: Joint Pareto classification using `Sales ($)` and `Hits` into classes A, B, C, and D with automated summary metrics and CSV export.

## How to Run
In your terminal:
```bash
cd C:\Users\joshu\.gemini\antigravity\scratch\inventory-decision-agent
pip install -r requirements.txt
streamlit run app.py
```
