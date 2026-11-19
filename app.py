"""
Inventory Decision Agent
An interactive decision support application for inventory analytics, optimization,
and multicriteria classification.

Retains all core inventory modules:
1. Classic Economic Order Quantity (EOQ)
2. All-Units Quantity Discount Analyzer
3. Safety Stock and Reorder Point (Continuous Review)
4. Fixed-Period Review Model (Periodic Review / Order-up-to Level)
5. Newsvendor Model (Single-Period Stochastic Demand)
6. Multicriteria Inventory Classification (Sales $ and Hits)
"""

import math
import streamlit as st
import pandas as pd
import numpy as np

# Import our analytical models
from inventory_models import (
    calculate_eoq,
    calculate_quantity_discounts,
    calculate_safety_stock_rop,
    calculate_fixed_period_review,
    calculate_newsvendor,
)
from classification import classify_inventory_multicriteria

# -----------------------------------------------------------------------------
# APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Inventory Decision Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📦 Inventory Decision Agent")
st.caption("Inventory Analytics, Optimization, & Multicriteria Classification Engine")

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION FOR CSV DATA
# -----------------------------------------------------------------------------
if "inventory_df" not in st.session_state:
    st.session_state["inventory_df"] = None

# -----------------------------------------------------------------------------
# SIDEBAR: NAVIGATION & GLOBAL CSV UPLOADER
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Data & Navigation")

uploaded_file = st.sidebar.file_uploader(
    "Upload Inventory CSV",
    type=["csv"],
    help="Upload your inventory dataset with Item Number, Item Name, UoM, Sales ($), and Hits.",
)

if uploaded_file is not None:
    try:
        # Load CSV using UTF-8 to prevent encoding errors
        df_uploaded = pd.read_csv(uploaded_file, encoding="utf-8")
        st.session_state["inventory_df"] = df_uploaded
        st.sidebar.success(f"Loaded: {uploaded_file.name} ({len(df_uploaded):,} items)")
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")

module = st.sidebar.radio(
    "Select Analysis Tool",
    [
        "1. Classic EOQ Calculator",
        "2. Quantity Discount Analysis",
        "3. Safety Stock & Reorder Point",
        "4. Fixed-Period Review Model",
        "5. Newsvendor Model",
        "6. Multicriteria Inventory Classification",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Course Note:** Ensure input units match (e.g., annual vs. daily rates) "
    "and cross-check answers with the Project Report Generator."
)

# =============================================================================
# MODULE 1: CLASSIC ECONOMIC ORDER QUANTITY (EOQ)
# =============================================================================
if module == "1. Classic EOQ Calculator":
    st.header("1. Economic Order Quantity (EOQ) Calculator")
    st.write(
        "Determines the optimal order quantity that minimizes the total annual cost of ordering and holding inventory."
    )

    col1, col2 = st.columns(2)
    with col1:
        annual_demand = st.number_input(
            "Annual Demand ($D$, units/year)",
            min_value=1.0,
            value=10000.0,
            step=500.0,
            format="%.0f",
        )
        order_cost = st.number_input(
            "Order / Setup Cost ($S$, $/order)",
            min_value=0.1,
            value=50.0,
            step=5.0,
            format="%.2f",
        )
        unit_cost = st.number_input(
            "Unit Purchase Cost ($C$, $/unit)",
            min_value=0.01,
            value=25.0,
            step=1.0,
            format="%.2f",
        )

    with col2:
        holding_input_type = st.radio(
            "Holding Cost Specification",
            ["Holding Cost Rate (% of Unit Cost)", "Direct Holding Cost ($/unit/year)"],
        )
        if holding_input_type == "Holding Cost Rate (% of Unit Cost)":
            holding_rate_pct = st.number_input(
                "Annual Holding Rate ($h$, %)",
                min_value=0.1,
                max_value=200.0,
                value=20.0,
                step=1.0,
            )
            holding_rate = holding_rate_pct / 100.0
            holding_cost_unit = None
        else:
            holding_cost_unit = st.number_input(
                "Holding Cost per Unit/Year ($H$, $/unit/year)",
                min_value=0.01,
                value=5.0,
                step=0.5,
            )
            holding_rate = None

        work_days = st.number_input(
            "Working Days per Year", min_value=1, max_value=366, value=250
        )
        lead_time = st.number_input(
            "Lead Time (days, optional for deterministic ROP)",
            min_value=0.0,
            value=5.0,
            step=1.0,
        )

    if st.button("Compute EOQ", type="primary"):
        res = calculate_eoq(
            annual_demand=annual_demand,
            order_cost=order_cost,
            unit_cost=unit_cost,
            holding_cost_rate=holding_rate,
            holding_cost_per_unit=holding_cost_unit,
            work_days_per_year=int(work_days),
            lead_time_days=lead_time,
        )

        st.subheader("Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Optimal EOQ ($Q^*$)", f"{res['eoq']:,.1f} units", help=f"Rounded up: {res['eoq_rounded']} units")
        m2.metric("Orders per Year ($N$)", f"{res['orders_per_year']:.2f}")
        m3.metric("Cycle Time", f"{res['cycle_time_days']:.1f} days")
        m4.metric("Holding Cost / Unit ($H$)", f"${res['holding_cost_per_unit']:.2f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annual Ordering Cost", f"${res['annual_ordering_cost']:,.2f}")
        c2.metric("Annual Holding Cost", f"${res['annual_holding_cost']:,.2f}")
        c3.metric("Total Inventory Cost", f"${res['total_inventory_cost']:,.2f}")
        c4.metric("Total Annual Cost (inc. purchase)", f"${res['total_annual_cost']:,.2f}")

        if lead_time > 0:
            st.info(f"**Deterministic Reorder Point (Lead Time Demand):** {res['lead_time_demand']:,.1f} units")

        # Visual Cost Curve
        q_vals = np.linspace(max(10, res["eoq"] * 0.2), res["eoq"] * 2.5, 100)
        h_unit = res["holding_cost_per_unit"]
        ordering_costs = (annual_demand / q_vals) * order_cost
        holding_costs = (q_vals / 2.0) * h_unit
        total_costs = ordering_costs + holding_costs

        chart_df = pd.DataFrame(
            {
                "Order Quantity (Q)": q_vals,
                "Annual Ordering Cost": ordering_costs,
                "Annual Holding Cost": holding_costs,
                "Total Inventory Cost": total_costs,
            }
        ).set_index("Order Quantity (Q)")

        st.line_chart(chart_df)


# =============================================================================
# MODULE 2: QUANTITY DISCOUNT MODEL
# =============================================================================
elif module == "2. Quantity Discount Analysis":
    st.header("2. All-Units Quantity Discount Analyzer")
    st.write(
        "Evaluates supplier price breaks to find the order quantity that minimizes total annual costs."
    )

    col1, col2 = st.columns(2)
    with col1:
        annual_demand_qd = st.number_input(
            "Annual Demand ($D$)", min_value=1.0, value=20000.0, step=1000.0
        )
        order_cost_qd = st.number_input(
            "Order Cost ($S$, $/order)", min_value=0.1, value=100.0, step=10.0
        )
    with col2:
        holding_rate_pct_qd = st.number_input(
            "Annual Holding Rate ($h$, %)", min_value=0.1, value=20.0, step=1.0
        )
        holding_rate_qd = holding_rate_pct_qd / 100.0

    st.subheader("Price Discount Tiers")
    st.write("Specify minimum quantity and price per unit for each tier.")

    default_tiers_df = pd.DataFrame(
        [
            {"Tier Name": "Tier 1", "Min Qty": 1, "Max Qty": 999, "Unit Price": 10.00},
            {"Tier Name": "Tier 2", "Min Qty": 1000, "Max Qty": 1999, "Unit Price": 9.50},
            {"Tier Name": "Tier 3", "Min Qty": 2000, "Max Qty": None, "Unit Price": 9.00},
        ]
    )

    edited_tiers = st.data_editor(
        default_tiers_df,
        num_rows="dynamic",
        use_container_width=True,
    )

    if st.button("Evaluate Quantity Discounts", type="primary"):
        tiers_list = []
        for _, r in edited_tiers.iterrows():
            max_q = None if pd.isna(r["Max Qty"]) or r["Max Qty"] in ["None", ""] else float(r["Max Qty"])
            tiers_list.append(
                {
                    "tier_name": str(r["Tier Name"]),
                    "min_qty": float(r["Min Qty"]),
                    "max_qty": max_q,
                    "unit_price": float(r["Unit Price"]),
                }
            )

        res = calculate_quantity_discounts(
            annual_demand=annual_demand_qd,
            order_cost=order_cost_qd,
            holding_cost_rate=holding_rate_qd,
            discount_tiers=tiers_list,
        )

        best = res["best_tier"]
        st.success(
            f"### Recommended Policy: Order **{math.ceil(res['optimal_order_quantity']):,} units** at **${best['unit_price']:.2f}/unit** ({best['tier_name']})"
        )
        st.metric("Minimum Total Annual Cost", f"${res['min_total_annual_cost']:,.2f}")

        res_table = []
        for t in res["all_tiers"]:
            res_table.append(
                {
                    "Tier": t["tier_name"],
                    "Unit Price ($)": f"${t['unit_price']:.2f}",
                    "Min Qty": f"{t['min_qty']:,.0f}",
                    "Raw EOQ": f"{t['raw_eoq']:,.1f}",
                    "Evaluated Qty": f"{math.ceil(t['evaluated_qty']):,.0f}",
                    "Feasibility / Status": t["status"],
                    "Purchase Cost": f"${t['annual_purchase_cost']:,.2f}",
                    "Ordering Cost": f"${t['annual_ordering_cost']:,.2f}",
                    "Holding Cost": f"${t['annual_holding_cost']:,.2f}",
                    "Total Annual Cost": f"${t['total_annual_cost']:,.2f}",
                }
            )

        st.table(pd.DataFrame(res_table))


# =============================================================================
# MODULE 3: SAFETY STOCK & REORDER POINT
# =============================================================================
elif module == "3. Safety Stock & Reorder Point":
    st.header("3. Safety Stock & Reorder Point (Continuous Review)")
    st.write(
        "Calculates the buffer stock and reorder point required to satisfy demand variability during replenishment lead time."
    )

    col1, col2 = st.columns(2)
    with col1:
        avg_d = st.number_input(
            "Average Demand per Period ($\bar{d}$)", min_value=0.1, value=50.0, step=5.0
        )
        std_d = st.number_input(
            "Demand Standard Deviation per Period ($\sigma_d$)",
            min_value=0.0,
            value=10.0,
            step=1.0,
        )
        csl = st.slider(
            "Desired Cycle Service Level ($CSL$)",
            min_value=0.50,
            max_value=0.999,
            value=0.95,
            step=0.01,
            format="%.3f",
        )

    with col2:
        lead_t = st.number_input(
            "Replenishment Lead Time ($L$ periods)", min_value=0.1, value=6.0, step=1.0
        )
        lead_time_variable = st.checkbox("Lead Time is Stochastic (Variable)?")
        if lead_time_variable:
            std_lt = st.number_input(
                "Lead Time Standard Deviation ($\sigma_L$)",
                min_value=0.1,
                value=1.5,
                step=0.5,
            )
        else:
            std_lt = 0.0

    if st.button("Compute Safety Stock & ROP", type="primary"):
        res = calculate_safety_stock_rop(
            avg_daily_demand=avg_d,
            std_daily_demand=std_d,
            lead_time_days=lead_t,
            cycle_service_level=csl,
            std_lead_time_days=std_lt,
        )

        st.subheader("Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Safety Stock ($SS$)", f"{res['safety_stock_rounded']:,} units", help=f"Exact: {res['safety_stock']:.2f}")
        m2.metric("Reorder Point ($ROP$)", f"{res['reorder_point_rounded']:,} units", help=f"Exact: {res['reorder_point']:.2f}")
        m3.metric("Lead Time Demand ($DDLT$)", f"{res['ddlt']:,.1f} units")
        m4.metric("Service Level Z-Score ($z$)", f"{res['z_score']:.3f}")

        st.info(
            f"**Policy Rule:** Place a replenishment order whenever inventory drops to or below **{res['reorder_point_rounded']:,} units**. "
            f"This maintains a **{csl * 100:.1f}%** cycle service level (risk of stockout per cycle: {res['stockout_risk'] * 100:.1f}%)."
        )


# =============================================================================
# MODULE 4: FIXED-PERIOD REVIEW MODEL
# =============================================================================
elif module == "4. Fixed-Period Review Model":
    st.header("4. Fixed-Period Review Model (Periodic Review / Order-Up-To Level)")
    st.write(
        "Determines target inventory level $M$ (Order-up-to level) over the protection interval $T + L$."
    )

    col1, col2 = st.columns(2)
    with col1:
        rev_period = st.number_input(
            "Review Period ($T$, days/periods)", min_value=1.0, value=7.0, step=1.0
        )
        lead_t_pr = st.number_input(
            "Lead Time ($L$, days/periods)", min_value=0.0, value=3.0, step=1.0
        )
        avg_d_pr = st.number_input(
            "Average Demand per Period ($\bar{d}$)", min_value=0.1, value=20.0, step=2.0
        )
        std_d_pr = st.number_input(
            "Demand Std Dev per Period ($\sigma_d$)", min_value=0.0, value=5.0, step=1.0
        )

    with col2:
        csl_pr = st.slider(
            "Target Cycle Service Level ($CSL$)",
            min_value=0.50,
            max_value=0.999,
            value=0.95,
            step=0.01,
        )
        on_hand = st.number_input(
            "Current On-Hand Inventory ($I$)", min_value=0.0, value=45.0, step=5.0
        )
        on_order = st.number_input(
            "Current On-Order / Open Orders ($IO$)", min_value=0.0, value=15.0, step=5.0
        )
        backorders = st.number_input(
            "Current Backorders ($BO$)", min_value=0.0, value=0.0, step=1.0
        )

    if st.button("Compute Periodic Review Policy", type="primary"):
        res = calculate_fixed_period_review(
            review_period_days=rev_period,
            lead_time_days=lead_t_pr,
            avg_daily_demand=avg_d_pr,
            std_daily_demand=std_d_pr,
            cycle_service_level=csl_pr,
            current_on_hand=on_hand,
            current_on_order=on_order,
            backorders=backorders,
        )

        st.subheader("Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Protection Interval ($T + L$)", f"{res['protection_interval_days']:.0f} periods")
        m2.metric("Target Order-Up-To Level ($M$)", f"{res['target_level_rounded']:,} units")
        m3.metric("Order Quantity to Place ($Q$)", f"{res['order_quantity_rounded']:,} units")

        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Protection Interval Demand", f"{res['expected_demand_pi']:.1f} units")
        c2.metric("Safety Stock ($SS$)", f"{res['safety_stock_rounded']:,} units")
        c3.metric("Current Inventory Position", f"{res['current_inventory_position']:.0f} units")


# =============================================================================
# MODULE 5: NEWSVENDOR MODEL
# =============================================================================
elif module == "5. Newsvendor Model":
    st.header("5. Newsvendor Model (Single-Period Stochastic Inventory)")
    st.write(
        "Optimizes order quantities for perishable or seasonal items facing uncertain demand."
    )

    col1, col2 = st.columns(2)
    with col1:
        price_nv = st.number_input(
            "Selling Price per Unit ($P$)", min_value=0.1, value=100.0, step=5.0
        )
        cost_nv = st.number_input(
            "Unit Purchase/Production Cost ($C$)", min_value=0.1, value=60.0, step=5.0
        )
        salvage_nv = st.number_input(
            "Salvage / Clearance Value per Unsold Unit ($S$)",
            min_value=0.0,
            value=20.0,
            step=5.0,
        )
        penalty_nv = st.number_input(
            "Stockout Goodwill Penalty ($g$, optional)", min_value=0.0, value=0.0, step=1.0
        )

    with col2:
        mean_d_nv = st.number_input(
            "Mean Demand ($\mu$)", min_value=1.0, value=1000.0, step=50.0
        )
        std_d_nv = st.number_input(
            "Demand Standard Deviation ($\sigma$)", min_value=0.1, value=200.0, step=10.0
        )

    if st.button("Run Newsvendor Optimization", type="primary"):
        res = calculate_newsvendor(
            selling_price=price_nv,
            unit_cost=cost_nv,
            salvage_value=salvage_nv,
            mean_demand=mean_d_nv,
            std_demand=std_d_nv,
            stockout_penalty=penalty_nv,
        )

        st.subheader("Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Underage Cost ($C_u$)", f"${res['underage_cost_cu']:,.2f}")
        m2.metric("Overage Cost ($C_o$)", f"${res['overage_cost_co']:,.2f}")
        m3.metric("Critical Fractile ($CR$)", f"{res['critical_ratio']:.4f}")
        m4.metric("Optimal Order Quantity ($Q^*$)", f"{res['optimal_quantity_rounded']:,} units")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimal Z-Score ($z^*$)", f"{res['z_star']:.3f}")
        c2.metric("Expected Leftover Units", f"{res['expected_leftover']:,.1f} units")
        c3.metric("Expected Lost Sales", f"{res['expected_lost_sales']:,.1f} units")
        c4.metric("Expected Profit", f"${res['expected_profit']:,.2f}")


# =============================================================================
# MODULE 6: MULTICRITERIA INVENTORY CLASSIFICATION
# =============================================================================
elif module == "6. Multicriteria Inventory Classification":
    st.header("6. Multicriteria Inventory Classification")
    st.write(
        "Performs joint inventory classification into Classes **A, B, C, and D** using both **Sales ($)** and **Hits** "
        "following the weighted scoring methodology (65% Sales + 35% Hits)."
    )

    # Uses the existing uploaded DataFrame from st.session_state (no duplicate uploader created)
    df_data = st.session_state["inventory_df"]

    if df_data is None:
        st.warning(
            "⚠️ No inventory file currently uploaded! Please upload `ProjectDataV2_UTF8.csv` in the **sidebar** to run classification."
        )
        st.info(
            "Expected columns: `Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`."
        )
    else:
        st.success(f"Connected to Uploaded Data: **{len(df_data):,}** inventory items.")

        # Verify required columns
        req_cols = ["Item Number", "Item Name", "UoM", "Sales ($)", "Hits"]
        missing_cols = [c for c in req_cols if c not in df_data.columns]

        if missing_cols:
            st.error(f"Missing required columns in uploaded CSV: {missing_cols}")
        else:
            with st.expander("ℹ️ Methodology & Classification Rules", expanded=False):
                st.markdown(
                    """
                    * **Single Criterion Cutoffs (unrounded cumulative %):**
                      * **A**: Through 60% ($\le 60\%$)
                      * **B**: Above 60% through 80% ($>60\%$ and $\le 80\%$)
                      * **C**: Above 80% through 90% ($>80\%$ and $\le 90\%$)
                      * **D**: Above 90% ($>90\%$)
                      * *Ties:* Keep original CSV order (stable sort).
                    * **Score Conversion:** $A = 4$, $B = 3$, $C = 2$, $D = 1$.
                    * **Combined Score:** $0.65 \times \text{Sales Score} + 0.35 \times \text{Hits Score}$.
                    * **Final Class Assignment:**
                      * **A**: $\ge 3.25$
                      * **B**: $2.50 \le \text{Score} < 3.25$ (lower bound inclusive, upper bound exclusive)
                      * **C**: $1.75 \le \text{Score} < 2.50$ (lower bound inclusive, upper bound exclusive)
                      * **D**: $< 1.75$
                    """
                )

            # Perform classification automatically
            df_result, summary_df = classify_inventory_multicriteria(
                df=df_data,
                item_col="Item Number",
                sales_col="Sales ($)",
                hits_col="Hits",
            )

            # Store in session state
            st.session_state["classified_df"] = df_result
            st.session_state["summary_df"] = summary_df

            # Top KPI metrics
            total_items = len(df_result)
            total_sales = df_result["Sales ($)"].sum()
            total_hits = df_result["Hits"].sum()

            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Total Inventory Items", f"{total_items:,}")
            kpi2.metric("Total Sales Volume", f"${total_sales:,.2f}")
            kpi3.metric("Total Warehouse Hits", f"{int(total_hits):,}")

            # Summary Table
            st.subheader("📊 Multicriteria Classification Summary")
            st.dataframe(
                summary_df.style.format(
                    {
                        "Item Count": "{:,}",
                        "% of Total Items": "{:.2f}%",
                        "Total Sales ($)": "${:,.2f}",
                        "% of Total Sales": "{:.2f}%",
                        "Total Hits": "{:,}",
                        "% of Total Hits": "{:.2f}%",
                    }
                ),
                use_container_width=True,
            )

            # Item-Level Detail Table
            st.subheader("📋 Classified Inventory Items")
            filter_class = st.multiselect(
                "Filter by Final Class:",
                options=["A", "B", "C", "D"],
                default=["A", "B", "C", "D"],
            )

            display_cols = [
                "Item Number",
                "Item Name",
                "UoM",
                "Sales ($)",
                "Sales_Cum_Pct",
                "Sales_Class",
                "Sales_Score",
                "Hits",
                "Hits_Cum_Pct",
                "Hits_Class",
                "Hits_Score",
                "Combined_Score",
                "Final_Class",
            ]

            filtered_df = df_result[df_result["Final_Class"].isin(filter_class)][display_cols]
            
            st.dataframe(
                filtered_df.style.format(
                    {
                        "Sales ($)": "${:,.2f}",
                        "Sales_Cum_Pct": "{:.2f}%",
                        "Hits": "{:,}",
                        "Hits_Cum_Pct": "{:.2f}%",
                        "Combined_Score": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

            # CSV Download Button
            csv_data = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Classified Dataset (CSV)",
                data=csv_data,
                file_name="ProjectDataV2_Classified.csv",
                mime="text/csv",
                type="primary",
            )
