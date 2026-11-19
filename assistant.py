"""
Inventory Decision Assistant Module
Provides natural language query parsing and response generation based on the inventory dataset.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional


def generate_inventory_response(prompt: str, df: Optional[pd.DataFrame], summary_df: Optional[pd.DataFrame]) -> str:
    """
    Generates an analytical, data-driven response to a user prompt based on the inventory dataset.
    """
    if df is None or len(df) == 0:
        return (
            "⚠️ **No inventory dataset is currently loaded.**\n\n"
            "Please upload your inventory CSV (e.g., `ProjectDataV2_UTF8.csv`) in the sidebar uploader first "
            "so I can analyze your inventory data."
        )

    p_lower = prompt.lower().strip()
    total_items = len(df)
    total_sales = df["Sales ($)"].sum() if "Sales ($)" in df.columns else 0.0
    total_hits = df["Hits"].sum() if "Hits" in df.columns else 0

    # -------------------------------------------------------------------------
    # 1. SUMMARY / OVERVIEW PROMPT
    # -------------------------------------------------------------------------
    if any(k in p_lower for k in ["overview", "summary", "summarize", "breakdown", "overall"]):
        if summary_df is not None:
            table_md = summary_df.to_markdown(index=False)
            return (
                f"### 📊 Inventory Overview & Classification Summary\n\n"
                f"**Total SKUs Analyzed:** {total_items:,}\n"
                f"**Total Annual Sales:** ${total_sales:,.2f}\n"
                f"**Total Warehouse Hits:** {int(total_hits):,}\n\n"
                f"#### Classification Distribution Table:\n\n{table_md}\n\n"
                f"**Key Insights:**\n"
                f"- **Class A items** represent the primary financial and operational drivers of your supply chain.\n"
                f"- **Class D items** represent slow-moving or low-value inventory that should be managed with lean reordering or evaluated for SKU rationalization."
            )

    # -------------------------------------------------------------------------
    # 2. CLASS-SPECIFIC QUERIES (Class A, B, C, or D)
    # -------------------------------------------------------------------------
    for target_class in ["a", "b", "c", "d"]:
        class_upper = target_class.upper()
        # Look for phrases like "class a", "category a", "show a", "items in a"
        if re.search(rf"\bclass\s+{target_class}\b", p_lower) or re.search(rf"\bcategory\s+{target_class}\b", p_lower):
            cls_df = df[df["Final_Class"] == class_upper]
            count = len(cls_df)
            sales = cls_df["Sales ($)"].sum()
            hits = cls_df["Hits"].sum()
            sales_pct = (sales / total_sales * 100.0) if total_sales > 0 else 0.0
            hits_pct = (hits / total_hits * 100.0) if total_hits > 0 else 0.0

            top_items = cls_df.sort_values(by="Sales ($)", ascending=False).head(5)
            top_items_list = ""
            for _, r in top_items.iterrows():
                top_items_list += f"- **Item {r['Item Number']}** ({r.get('Item Name', 'N/A')}): ${r['Sales ($)']:,.2f} sales, {int(r['Hits']):,} hits\n"

            policy_recommendations = {
                "A": "Continuous review policy (ROP, Q) with high Cycle Service Level (e.g. 98%), weekly cycle counting, and priority supplier management.",
                "B": "Periodic review or standard ROP policy with 95% service level and monthly cycle counts.",
                "C": "Simplified periodic review or two-bin replenishment with moderate safety stock to avoid stockout administrative overhead.",
                "D": "Minimum-maximum replenishment or make-to-order; scrutinize for potential obsolescence and consider inventory run-out or disposal.",
            }

            return (
                f"### 🏷️ Analysis for Class {class_upper} Inventory\n\n"
                f"- **Total Item Count:** {count:,} items ({count / total_items * 100.0:.2f}% of catalog)\n"
                f"- **Total Sales Volume:** ${sales:,.2f} ({sales_pct:.2f}% of total sales)\n"
                f"- **Total Warehouse Hits:** {int(hits):,} ({hits_pct:.2f}% of total hits)\n\n"
                f"#### Top Items in Class {class_upper}:\n"
                f"{top_items_list}\n"
                f"#### Recommended Inventory Policy for Class {class_upper}:\n"
                f"{policy_recommendations.get(class_upper, 'Standard inventory monitoring.')}"
            )

    # -------------------------------------------------------------------------
    # 3. DISCREPANCY / ANOMALY QUERIES (High Hits / Low Sales or Vice Versa)
    # -------------------------------------------------------------------------
    if any(k in p_lower for k in ["high hit", "low sale", "fast mover", "workload", "warehouse layout"]):
        # Hits Class A or B, but Sales Class C or D
        high_hits_low_sales = df[
            (df["Hits_Class"].isin(["A", "B"])) & (df["Sales_Class"].isin(["C", "D"]))
        ]
        count = len(high_hits_low_sales)
        sample = high_hits_low_sales.head(5)
        sample_md = ""
        for _, r in sample.iterrows():
            sample_md += f"- **Item {r['Item Number']}** ({r.get('Item Name', 'N/A')}): Hits Class {r['Hits_Class']} ({int(r['Hits']):,} hits) vs Sales Class {r['Sales_Class']} (${r['Sales ($)']:,.2f})\n"

        return (
            f"### 📦 High Pick Frequency ('Hits') with Low Dollar Sales\n\n"
            f"Found **{count} items** that generate substantial warehouse movement despite low dollar sales volume.\n\n"
            f"#### Why this matters:\n"
            f"Under traditional single-criterion ABC analysis (Sales only), these items would be classified as Class C or D and relegated to the back of the warehouse. "
            f"However, because their **Hits are high**, workers must travel to pick them frequently, driving up labor and picking costs.\n\n"
            f"#### Key Items Identified:\n{sample_md}\n"
            f"#### Strategic Recommendation:\n"
            f"Slot these items in the **golden zone** (at waist height, close to packing and shipping docks) to minimize picking transit time."
        )

    if any(k in p_lower for k in ["high sale", "low hit", "expensive", "slow moving high value"]):
        # Sales Class A or B, but Hits Class C or D
        high_sales_low_hits = df[
            (df["Sales_Class"].isin(["A", "B"])) & (df["Hits_Class"].isin(["C", "D"]))
        ]
        count = len(high_sales_low_hits)
        sample = high_sales_low_hits.head(5)
        sample_md = ""
        for _, r in sample.iterrows():
            sample_md += f"- **Item {r['Item Number']}** ({r.get('Item Name', 'N/A')}): Sales Class {r['Sales_Class']} (${r['Sales ($)']:,.2f}) vs Hits Class {r['Hits_Class']} ({int(r['Hits']):,} hits)\n"

        return (
            f"### 💰 High Dollar Sales with Low Pick Frequency\n\n"
            f"Found **{count} items** that represent significant sales revenue but are ordered infrequently.\n\n"
            f"#### Key Items Identified:\n{sample_md}\n"
            f"#### Strategic Recommendation:\n"
            f"Keep tight inventory controls and low safety stocks to avoid tying up capital in expensive inventory. "
            f"Store in secure, standard storage locations rather than prime picking lanes."
        )

    # -------------------------------------------------------------------------
    # 4. POLICY / RECOMMENDATIONS PROMPT
    # -------------------------------------------------------------------------
    if any(k in p_lower for k in ["policy", "recommend", "strategy", "action", "guideline"]):
        return (
            f"### 📋 Strategic Inventory Management Recommendations\n\n"
            f"Based on your dataset's multicriteria classification (65% Sales + 35% Hits):\n\n"
            f"1. **Class A Items (Critical / Core Assets - Score $\\ge 3.25$):**\n"
            f"   - **Replenishment:** Continuous review $(ROP, Q)$ using exact EOQ lot sizes.\n"
            f"   - **Service Level:** 98%–99% cycle service level.\n"
            f"   - **Control:** Weekly physical cycle counts and tight supplier collaboration to reduce lead time variability.\n\n"
            f"2. **Class B Items (Moderate Value & Volume - Score $2.50 \\le x < 3.25$):**\n"
            f"   - **Replenishment:** Periodic review (e.g. bi-weekly $(T, M)$) or standard ROP.\n"
            f"   - **Service Level:** 95% cycle service level.\n"
            f"   - **Control:** Monthly cycle counts and standard automated ordering.\n\n"
            f"3. **Class C Items (Low-Impact - Score $1.75 \\le x < 2.50$):**\n"
            f"   - **Replenishment:** Two-bin system or bulk orders to minimize order processing overhead.\n"
            f"   - **Service Level:** 90% service level.\n"
            f"   - **Control:** Quarterly counts.\n\n"
            f"4. **Class D Items (Long-Tail / Obsolete Candidates - Score $< 1.75$):**\n"
            f"   - **Replenishment:** Made-to-order or lean min-max replenishment with high thresholds.\n"
            f"   - **Action:** Evaluate for SKU rationalization, liquidation, or vendor-managed inventory."
        )

    # -------------------------------------------------------------------------
    # 5. SPECIFIC ITEM SEARCH (e.g. by Item Number)
    # -------------------------------------------------------------------------
    # Check if a number in prompt matches an Item Number
    item_matches = [w for w in re.findall(r"\b\w+\b", prompt) if w in df["Item Number"].astype(str).values]
    if item_matches:
        target_item = item_matches[0]
        item_row = df[df["Item Number"].astype(str) == target_item].iloc[0]
        return (
            f"### 🔍 Item Detail: {target_item}\n\n"
            f"- **Item Name:** {item_row.get('Item Name', 'N/A')}\n"
            f"- **Unit of Measure (UoM):** {item_row.get('UoM', 'N/A')}\n"
            f"- **Sales ($):** ${item_row['Sales ($)']:,.2f} (Cumulative: {item_row.get('Sales_Cum_Pct', 0):.2f}% | Class {item_row.get('Sales_Class', 'N/A')})\n"
            f"- **Hits:** {int(item_row['Hits']):,} (Cumulative: {item_row.get('Hits_Cum_Pct', 0):.2f}% | Class {item_row.get('Hits_Class', 'N/A')})\n"
            f"- **Sales Score:** {item_row.get('Sales_Score', 'N/A')} (Weight: 65%)\n"
            f"- **Hits Score:** {item_row.get('Hits_Score', 'N/A')} (Weight: 35%)\n"
            f"- **Combined Score:** {item_row.get('Combined_Score', 0):.2f}\n"
            f"- **Final Classification:** **Class {item_row.get('Final_Class', 'N/A')}**"
        )

    # -------------------------------------------------------------------------
    # 6. DEFAULT INTELLIGENT SYNTHESIS
    # -------------------------------------------------------------------------
    # Return general analytical breakdown answering their prompt
    return (
        f"### 💡 Inventory Analytics Agent Response\n\n"
        f"**Query Received:** *\"{prompt}\"*\n\n"
        f"**Dataset Analysis Summary:**\n"
        f"- **Dataset Size:** {total_items:,} items\n"
        f"- **Total Sales Volume:** ${total_sales:,.2f}\n"
        f"- **Total Warehouse Hits:** {int(total_hits):,}\n\n"
        f"**Classification Breakdown (65% Sales + 35% Hits):**\n"
        f"- **Class A:** {len(df[df['Final_Class']=='A']):,} items (Highest strategic priority, combined score $\ge 3.25$)\n"
        f"- **Class B:** {len(df[df['Final_Class']=='B']):,} items (Moderate volume, score $2.50 \le x < 3.25$)\n"
        f"- **Class C:** {len(df[df['Final_Class']=='C']):,} items (Low-moderate volume, score $1.75 \le x < 2.50$)\n"
        f"- **Class D:** {len(df[df['Final_Class']=='D']):,} items (Long-tail / slow moving, score $< 1.75$)\n\n"
        f"You can ask me specific questions such as:\n"
        f"- *\"Show Class A items\"*\n"
        f"- *\"What is the summary of Class D?\"*\n"
        f"- *\"Which items have high hits but low sales?\"*\n"
        f"- *\"What are our inventory policy recommendations?\"*\n"
        f"- *\"Lookup item 101\"*"
    )
