"""
Multicriteria Inventory Classification Module
Implements the exact UNCW SCM 517 multicriteria classification methodology:
1. Separate ranking of Sales ($) and Hits (highest to lowest, stable sort preserving original CSV order for ties).
2. Unrounded cumulative percentages:
   - A: <= 60%
   - B: > 60% and <= 80%
   - C: > 80% and <= 90%
   - D: > 90%
3. Match classifications by Item Number.
4. Convert classes to scores: A=4, B=3, C=2, D=1.
5. Combined Score = (0.65 * Sales_Score) + (0.35 * Hits_Score).
6. Final Class assignment:
   - A: Combined Score >= 3.25
   - B: 2.50 <= Combined Score < 3.25 (lower bound inclusive, upper bound exclusive)
   - C: 1.75 <= Combined Score < 2.50 (lower bound inclusive, upper bound exclusive)
   - D: Combined Score < 1.75
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def assign_single_criterion_tier(cum_pct: float) -> str:
    """
    Assigns A, B, C, or D based on unrounded cumulative percentage:
    - A: <= 60%
    - B: > 60% through 80%
    - C: > 80% through 90%
    - D: > 90%
    """
    if cum_pct <= 60.0:
        return "A"
    elif cum_pct <= 80.0:
        return "B"
    elif cum_pct <= 90.0:
        return "C"
    else:
        return "D"


def assign_final_class(combined_score: float) -> str:
    """
    Assigns Final Class based on combined weighted score:
    - A: >= 3.25
    - B: 2.50 <= score < 3.25 (inclusive lower, exclusive upper)
    - C: 1.75 <= score < 2.50 (inclusive lower, exclusive upper)
    - D: < 1.75
    """
    if combined_score >= 3.25:
        return "A"
    elif combined_score >= 2.50:
        return "B"
    elif combined_score >= 1.75:
        return "C"
    else:
        return "D"


def classify_inventory_multicriteria(
    df: pd.DataFrame,
    item_col: str = "Item Number",
    sales_col: str = "Sales ($)",
    hits_col: str = "Hits",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs multicriteria inventory classification on the uploaded DataFrame.
    
    Returns:
    - df_result: Complete DataFrame with all calculation steps, scores, and final classes.
    - summary_df: Summary metrics per Final Class (Item count, % Items, Total Sales, % Sales, Total Hits, % Hits).
    """
    SCORE_MAP = {"A": 4, "B": 3, "C": 2, "D": 1}

    # Work on a copy with original row order index preserved
    df_work = df.copy()
    df_work["_orig_order"] = np.arange(len(df_work))

    # Clean numeric columns
    df_work[sales_col] = pd.to_numeric(df_work[sales_col], errors="coerce").fillna(0.0)
    df_work[hits_col] = pd.to_numeric(df_work[hits_col], errors="coerce").fillna(0.0)

    # -------------------------------------------------------------------------
    # 1. SALES CLASSIFICATION
    # Rank highest to lowest. Keep original CSV order for ties (kind='mergesort' / stable).
    # -------------------------------------------------------------------------
    total_sales = df_work[sales_col].sum()
    df_sales_sorted = df_work.sort_values(
        by=sales_col, ascending=False, kind="mergesort"
    ).copy()
    
    if total_sales > 0:
        df_sales_sorted["Sales_Cum_Pct"] = (
            df_sales_sorted[sales_col].cumsum() / total_sales
        ) * 100.0
    else:
        df_sales_sorted["Sales_Cum_Pct"] = 100.0

    df_sales_sorted["Sales_Class"] = df_sales_sorted["Sales_Cum_Pct"].apply(
        assign_single_criterion_tier
    )
    df_sales_sorted["Sales_Score"] = df_sales_sorted["Sales_Class"].map(SCORE_MAP)

    # -------------------------------------------------------------------------
    # 2. HITS CLASSIFICATION
    # Rank highest to lowest. Keep original CSV order for ties (kind='mergesort' / stable).
    # -------------------------------------------------------------------------
    total_hits = df_work[hits_col].sum()
    df_hits_sorted = df_work.sort_values(
        by=hits_col, ascending=False, kind="mergesort"
    ).copy()
    
    if total_hits > 0:
        df_hits_sorted["Hits_Cum_Pct"] = (
            df_hits_sorted[hits_col].cumsum() / total_hits
        ) * 100.0
    else:
        df_hits_sorted["Hits_Cum_Pct"] = 100.0

    df_hits_sorted["Hits_Class"] = df_hits_sorted["Hits_Cum_Pct"].apply(
        assign_single_criterion_tier
    )
    df_hits_sorted["Hits_Score"] = df_hits_sorted["Hits_Class"].map(SCORE_MAP)

    # -------------------------------------------------------------------------
    # 3. MATCH BY ITEM NUMBER & COMBINE SCORES
    # -------------------------------------------------------------------------
    # Extract sales attributes
    sales_subset = df_sales_sorted[
        [item_col, "Sales_Cum_Pct", "Sales_Class", "Sales_Score"]
    ]
    
    # Extract hits attributes
    hits_subset = df_hits_sorted[
        [item_col, "Hits_Cum_Pct", "Hits_Class", "Hits_Score"]
    ]

    # Merge back to original DataFrame to maintain original order
    df_merged = df_work.merge(sales_subset, on=item_col, how="left")
    df_merged = df_merged.merge(hits_subset, on=item_col, how="left")

    # Restore exact original row order
    df_merged = df_merged.sort_values(by="_orig_order").drop(columns=["_orig_order"])

    # 4. Calculate Combined Score: 65% Sales Score + 35% Hits Score
    df_merged["Combined_Score"] = (
        0.65 * df_merged["Sales_Score"] + 0.35 * df_merged["Hits_Score"]
    )

    # 5. Assign Final Class
    df_merged["Final_Class"] = df_merged["Combined_Score"].apply(assign_final_class)

    # -------------------------------------------------------------------------
    # 6. GENERATE SUMMARY TABLE
    # Metrics per class: Item count, % items, Total Sales, % Sales, Total Hits, % Hits
    # -------------------------------------------------------------------------
    total_items = len(df_merged)
    summary_rows = []
    classes = ["A", "B", "C", "D"]

    for cls in classes:
        cls_df = df_merged[df_merged["Final_Class"] == cls]
        item_count = len(cls_df)
        item_pct = (item_count / total_items * 100.0) if total_items > 0 else 0.0
        cls_sales = cls_df[sales_col].sum()
        sales_pct = (cls_sales / total_sales * 100.0) if total_sales > 0 else 0.0
        cls_hits = cls_df[hits_col].sum()
        hits_pct = (cls_hits / total_hits * 100.0) if total_hits > 0 else 0.0

        summary_rows.append({
            "Class": cls,
            "Item Count": item_count,
            "% of Total Items": round(item_pct, 2),
            "Total Sales ($)": round(cls_sales, 2),
            "% of Total Sales": round(sales_pct, 2),
            "Total Hits": int(cls_hits),
            "% of Total Hits": round(hits_pct, 2),
        })

    # Add Total / Summary Row
    summary_rows.append({
        "Class": "Total",
        "Item Count": total_items,
        "% of Total Items": 100.0,
        "Total Sales ($)": round(total_sales, 2),
        "% of Total Sales": 100.0,
        "Total Hits": int(total_hits),
        "% of Total Hits": 100.0,
    })

    summary_df = pd.DataFrame(summary_rows)
    return df_merged, summary_df
