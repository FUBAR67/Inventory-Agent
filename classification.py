"""
Multicriteria Inventory Classification Module
Performs ABC / ABCD inventory classification using both 'Sales ($)' and 'Hits'.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional


def perform_single_criterion_classification(
    df: pd.DataFrame,
    metric_col: str,
    cutoffs: Dict[str, float] = None,
    ascending: bool = False,
    class_col_name: str = "Class",
) -> pd.DataFrame:
    """
    Classifies inventory items based on a single metric using cumulative percentage.
    
    Parameters:
    - df: Input DataFrame
    - metric_col: Column to rank by ('Sales ($)' or 'Hits')
    - cutoffs: Dict mapping Class to cumulative percentage threshold (e.g., {'A': 0.80, 'B': 0.95, 'C': 0.99, 'D': 1.00})
    - class_col_name: Name of the resulting class column
    """
    if cutoffs is None:
        cutoffs = {"A": 0.80, "B": 0.95, "C": 0.99, "D": 1.00}

    df_out = df.copy()
    
    # Ensure metric is numeric
    df_out[metric_col] = pd.to_numeric(df_out[metric_col], errors="coerce").fillna(0.0)
    
    # Sort descending
    df_out = df_out.sort_values(by=metric_col, ascending=ascending)
    
    total_val = df_out[metric_col].sum()
    if total_val == 0:
        df_out[f"{metric_col} %"] = 0.0
        df_out[f"{metric_col} Cum %"] = 0.0
        df_out[class_col_name] = "D"
        return df_out

    # Calculate item percentage and cumulative percentage
    df_out[f"{metric_col} %"] = (df_out[metric_col] / total_val) * 100.0
    df_out[f"{metric_col} Cum %"] = df_out[f"{metric_col} %"].cumsum()

    # Classification logic based on cumulative thresholds
    # Note: Handles tie-breaking and edge cases
    sorted_tiers = sorted(cutoffs.items(), key=lambda x: x[1])
    
    def assign_class(cum_pct):
        # cum_pct is 0 to 100
        for cls_name, threshold in sorted_tiers:
            # threshold can be passed as fraction (0.80) or percentage (80.0)
            t_val = threshold * 100.0 if threshold <= 1.0 else threshold
            if cum_pct <= t_val:
                return cls_name
        return sorted_tiers[-1][0]

    df_out[class_col_name] = df_out[f"{metric_col} Cum %"].apply(assign_class)
    return df_out


def default_multicriteria_matrix() -> Dict[Tuple[str, str], str]:
    """
    Standard Supply Chain 2D Decision Matrix mapping (Sales Class, Hits Class) -> Final Class (A, B, C, D).
    
    Sales \\ Hits |   A   |   B   |   C   |   D   |
    -------------|-------|-------|-------|-------|
          A      |   A   |   A   |   B   |   B   |
          B      |   A   |   B   |   B   |   C   |
          C      |   B   |   B   |   C   |   D   |
          D      |   B   |   C   |   D   |   D   |
    """
    matrix = {
        ("A", "A"): "A",
        ("A", "B"): "A",
        ("A", "C"): "B",
        ("A", "D"): "B",
        
        ("B", "A"): "A",
        ("B", "B"): "B",
        ("B", "C"): "B",
        ("B", "D"): "C",
        
        ("C", "A"): "B",
        ("C", "B"): "B",
        ("C", "C"): "C",
        ("C", "D"): "D",
        
        ("D", "A"): "B",
        ("D", "B"): "C",
        ("D", "C"): "D",
        ("D", "D"): "D",
    }
    return matrix


def classify_multicriteria(
    df: pd.DataFrame,
    sales_col: str = "Sales ($)",
    hits_col: str = "Hits",
    sales_cutoffs: Optional[Dict[str, float]] = None,
    hits_cutoffs: Optional[Dict[str, float]] = None,
    matrix_map: Optional[Dict[Tuple[str, str], str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs multicriteria inventory classification using Sales ($) and Hits.
    
    Returns:
    - df_classified: Original DataFrame with individual and composite classification columns added.
    - summary_df: Summary metrics per final class (Item count, % Items, Total Sales, % Sales, Total Hits, % Hits).
    """
    if sales_cutoffs is None:
        sales_cutoffs = {"A": 0.80, "B": 0.95, "C": 0.99, "D": 1.00}
    if hits_cutoffs is None:
        hits_cutoffs = {"A": 0.80, "B": 0.95, "C": 0.99, "D": 1.00}
    if matrix_map is None:
        matrix_map = default_multicriteria_matrix()

    # Step 1: Sales classification
    df_sales = perform_single_criterion_classification(
        df, metric_col=sales_col, cutoffs=sales_cutoffs, class_col_name="Sales_Class"
    )
    
    # Step 2: Hits classification
    df_hits = perform_single_criterion_classification(
        df, metric_col=hits_col, cutoffs=hits_cutoffs, class_col_name="Hits_Class"
    )

    # Merge or align
    df_classified = df.copy()
    df_classified["Sales_Class"] = df_sales["Sales_Class"]
    df_classified["Sales_Cum_Pct"] = df_sales[f"{sales_col} Cum %"].round(2)
    df_classified["Hits_Class"] = df_hits["Hits_Class"]
    df_classified["Hits_Cum_Pct"] = df_hits[f"{hits_col} Cum %"].round(2)

    # Step 3: Composite Multicriteria Class
    def map_composite(row):
        s_cls = str(row["Sales_Class"]).strip().upper()
        h_cls = str(row["Hits_Class"]).strip().upper()
        return matrix_map.get((s_cls, h_cls), "C")

    df_classified["Composite_Class"] = df_classified.apply(map_composite, axis=1)

    # Generate Summary Table
    total_items = len(df_classified)
    total_sales = df_classified[sales_col].sum()
    total_hits = df_classified[hits_col].sum()

    summary_rows = []
    classes = ["A", "B", "C", "D"]
    for cls in classes:
        cls_df = df_classified[df_classified["Composite_Class"] == cls]
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

    summary_df = pd.DataFrame(summary_rows)
    return df_classified, summary_df
