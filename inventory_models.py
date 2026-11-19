"""
Inventory Decision Agent - Core Analytical Models
Provides verified, analytically sound functions for:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Analysis
3. Safety Stock and Reorder Point (Continuous Review)
4. Fixed-Period Review Model (Periodic Review / Order-up-to Level)
5. Newsvendor Model (Single-Period Stochastic Demand)
"""

import math
from typing import Dict, List, Any, Optional
import scipy.stats as stats
import numpy as np


# =====================================================================
# 1. CLASSIC ECONOMIC ORDER QUANTITY (EOQ)
# =====================================================================
def calculate_eoq(
    annual_demand: float,
    order_cost: float,
    unit_cost: float,
    holding_cost_rate: Optional[float] = None,
    holding_cost_per_unit: Optional[float] = None,
    work_days_per_year: int = 250,
    lead_time_days: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes Classic EOQ and associated cost breakdowns.
    
    Parameters:
    - annual_demand (D): Annual demand in units
    - order_cost (S): Fixed cost per order
    - unit_cost (C): Purchase cost per unit
    - holding_cost_rate (h): Annual holding cost as decimal of unit cost (e.g., 0.20 = 20%)
    - holding_cost_per_unit (H): Annual holding cost per unit per year. If provided, overrides h * C.
    """
    if annual_demand <= 0 or order_cost < 0 or unit_cost <= 0:
        raise ValueError("Annual demand, order cost, and unit cost must be positive.")

    # Determine unit annual holding cost H
    if holding_cost_per_unit is not None and holding_cost_per_unit > 0:
        H = holding_cost_per_unit
        h_rate = H / unit_cost
    elif holding_cost_rate is not None and holding_cost_rate > 0:
        h_rate = holding_cost_rate
        H = h_rate * unit_cost
    else:
        raise ValueError("Must provide either a positive holding_cost_rate or holding_cost_per_unit.")

    # EOQ formula: Q* = sqrt(2 * D * S / H)
    eoq = math.sqrt((2 * annual_demand * order_cost) / H)
    
    # Orders per year
    orders_per_year = annual_demand / eoq
    
    # Cycle time in days
    cycle_time_days = (eoq / annual_demand) * work_days_per_year
    
    # Costs
    annual_ordering_cost = orders_per_year * order_cost
    annual_holding_cost = (eoq / 2.0) * H
    total_inventory_cost = annual_ordering_cost + annual_holding_cost
    annual_purchase_cost = annual_demand * unit_cost
    total_annual_cost = total_inventory_cost + annual_purchase_cost
    
    # Lead time demand / Reorder point (deterministic)
    daily_demand = annual_demand / work_days_per_year
    lead_time_demand = daily_demand * lead_time_days

    return {
        "eoq": eoq,
        "eoq_rounded": math.ceil(eoq),
        "orders_per_year": orders_per_year,
        "cycle_time_days": cycle_time_days,
        "holding_cost_per_unit": H,
        "holding_cost_rate": h_rate,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "total_inventory_cost": total_inventory_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "total_annual_cost": total_annual_cost,
        "lead_time_demand": lead_time_demand,
    }


# =====================================================================
# 2. QUANTITY DISCOUNT MODEL
# =====================================================================
def calculate_quantity_discounts(
    annual_demand: float,
    order_cost: float,
    holding_cost_rate: float,
    discount_tiers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates All-Units Quantity Discount pricing structures.
    
    discount_tiers: list of dicts with:
    - 'tier_name': str, e.g. "Tier 1"
    - 'min_qty': float, minimum order quantity
    - 'max_qty': float or None, maximum order quantity (None for infinity)
    - 'unit_price': float, price per unit
    """
    if annual_demand <= 0 or order_cost < 0 or holding_cost_rate <= 0:
        raise ValueError("Demand, order cost, and holding cost rate must be positive.")

    evaluated_tiers = []
    best_tier = None
    min_total_cost = float("inf")

    for tier in discount_tiers:
        p = float(tier["unit_price"])
        min_q = float(tier["min_qty"])
        max_q = float(tier["max_qty"]) if tier.get("max_qty") is not None else float("inf")
        
        # Unit holding cost for this tier
        H_i = holding_cost_rate * p
        
        # Unconstrained EOQ for this tier price
        eoq_raw = math.sqrt((2 * annual_demand * order_cost) / H_i)
        
        # Feasibility check
        is_feasible = (min_q <= eoq_raw <= max_q)
        
        if is_feasible:
            eval_q = eoq_raw
            status = "Feasible (EOQ within tier)"
        elif eoq_raw < min_q:
            eval_q = min_q
            status = f"Adjusted to price-break minimum ({min_q:,.0f})"
        else:
            eval_q = max_q
            status = "Above tier range (Superseded)"

        # Costs at evaluated quantity
        ordering_cost = (annual_demand / eval_q) * order_cost
        holding_cost = (eval_q / 2.0) * H_i
        purchase_cost = annual_demand * p
        total_cost = ordering_cost + holding_cost + purchase_cost

        tier_result = {
            "tier_name": tier.get("tier_name", f"Tier (min {min_q})"),
            "unit_price": p,
            "min_qty": min_q,
            "max_qty": max_q,
            "raw_eoq": eoq_raw,
            "evaluated_qty": eval_q,
            "status": status,
            "annual_purchase_cost": purchase_cost,
            "annual_ordering_cost": ordering_cost,
            "annual_holding_cost": holding_cost,
            "total_inventory_cost": ordering_cost + holding_cost,
            "total_annual_cost": total_cost,
        }
        evaluated_tiers.append(tier_result)

        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_tier = tier_result

    return {
        "best_tier": best_tier,
        "optimal_order_quantity": best_tier["evaluated_qty"],
        "min_total_annual_cost": min_total_cost,
        "all_tiers": evaluated_tiers,
    }


# =====================================================================
# 3. SAFETY STOCK & REORDER POINT (CONTINUOUS REVIEW)
# =====================================================================
def calculate_safety_stock_rop(
    avg_daily_demand: float,
    std_daily_demand: float,
    lead_time_days: float,
    cycle_service_level: float,
    std_lead_time_days: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes Safety Stock and Reorder Point (ROP) for a continuous review system.
    Supports constant lead time (std_lead_time_days = 0) or stochastic lead time.
    """
    if avg_daily_demand < 0 or std_daily_demand < 0 or lead_time_days < 0:
        raise ValueError("Demand, std dev, and lead time must be non-negative.")
    if not (0.50 < cycle_service_level < 1.0):
        raise ValueError("Cycle service level must be between 0.50 and 1.00.")

    # Expected Demand During Lead Time (DDLT)
    ddlt = avg_daily_demand * lead_time_days
    
    # Standard deviation of demand during lead time
    if std_lead_time_days > 0:
        # Both demand and lead time vary
        sigma_dlt = math.sqrt(
            (lead_time_days * (std_daily_demand ** 2)) +
            ((avg_daily_demand ** 2) * (std_lead_time_days ** 2))
        )
    else:
        # Constant lead time
        sigma_dlt = math.sqrt(lead_time_days) * std_daily_demand

    # Z-score corresponding to desired CSL
    z_score = float(stats.norm.ppf(cycle_service_level))
    
    # Safety stock
    safety_stock = z_score * sigma_dlt
    
    # Reorder point
    rop = ddlt + safety_stock
    
    # Stockout risk per cycle
    stockout_risk = 1.0 - cycle_service_level

    return {
        "ddlt": ddlt,
        "sigma_dlt": sigma_dlt,
        "z_score": z_score,
        "safety_stock": safety_stock,
        "safety_stock_rounded": math.ceil(safety_stock),
        "reorder_point": rop,
        "reorder_point_rounded": math.ceil(rop),
        "cycle_service_level": cycle_service_level,
        "stockout_risk": stockout_risk,
    }


# =====================================================================
# 4. FIXED-PERIOD REVIEW MODEL (PERIODIC REVIEW / ORDER-UP-TO LEVEL)
# =====================================================================
def calculate_fixed_period_review(
    review_period_days: float,
    lead_time_days: float,
    avg_daily_demand: float,
    std_daily_demand: float,
    cycle_service_level: float,
    current_on_hand: float = 0.0,
    current_on_order: float = 0.0,
    backorders: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes Target Order-Up-To Level (M or S) and recommended order quantity
    under periodic review (P, T).
    Protection Interval = Review Period (T) + Lead Time (L)
    """
    if review_period_days <= 0 or lead_time_days < 0 or avg_daily_demand < 0:
        raise ValueError("Review period must be positive; lead time and demand non-negative.")

    protection_interval = review_period_days + lead_time_days
    
    # Expected demand over protection interval
    expected_demand_pi = avg_daily_demand * protection_interval
    
    # Std dev of demand over protection interval
    sigma_pi = math.sqrt(protection_interval) * std_daily_demand
    
    # Z-score and safety stock
    z_score = float(stats.norm.ppf(cycle_service_level))
    safety_stock = z_score * sigma_pi
    
    # Target Order-Up-To Level (M)
    target_inventory_level = expected_demand_pi + safety_stock
    
    # Current inventory position
    inventory_position = current_on_hand + current_on_order - backorders
    
    # Order quantity required
    order_quantity = max(0.0, target_inventory_level - inventory_position)

    return {
        "protection_interval_days": protection_interval,
        "expected_demand_pi": expected_demand_pi,
        "sigma_pi": sigma_pi,
        "z_score": z_score,
        "safety_stock": safety_stock,
        "safety_stock_rounded": math.ceil(safety_stock),
        "target_order_up_to_level": target_inventory_level,
        "target_level_rounded": math.ceil(target_inventory_level),
        "current_inventory_position": inventory_position,
        "recommended_order_quantity": order_quantity,
        "order_quantity_rounded": math.ceil(order_quantity),
    }


# =====================================================================
# 5. NEWSVENDOR MODEL (SINGLE-PERIOD STOCHASTIC DEMAND)
# =====================================================================
def calculate_newsvendor(
    selling_price: float,
    unit_cost: float,
    salvage_value: float,
    mean_demand: float,
    std_demand: float,
    stockout_penalty: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes optimal order quantity and expected performance for single-period inventory.
    Assuming normal distribution of demand.
    """
    if selling_price <= unit_cost:
        raise ValueError("Selling price must exceed unit cost.")
    if salvage_value >= unit_cost:
        raise ValueError("Salvage value must be strictly less than unit cost.")
    if std_demand <= 0:
        raise ValueError("Demand standard deviation must be positive.")

    # Underage Cost Cu (margin on lost sale + penalty)
    cu = selling_price - unit_cost + stockout_penalty
    
    # Overage Cost Co (loss on unsold unit)
    co = unit_cost - salvage_value
    
    # Critical Fractile / Critical Ratio
    critical_ratio = cu / (cu + co)
    
    # Optimal z-score
    z_star = float(stats.norm.ppf(critical_ratio))
    
    # Optimal Order Quantity Q*
    optimal_q = mean_demand + z_star * std_demand
    
    # Standard normal calculations at z*
    phi_z = float(stats.norm.pdf(z_star))
    Phi_z = float(stats.norm.cdf(z_star))
    
    # Expected Leftover Units (Excess inventory)
    # E[Leftover] = (Q - mu)*Phi(z) + sigma*phi(z)
    expected_leftover = (optimal_q - mean_demand) * Phi_z + std_demand * phi_z
    
    # Expected Lost Sales
    # E[Lost Sales] = (mu - Q)*(1 - Phi(z)) + sigma*phi(z)
    expected_lost_sales = (mean_demand - optimal_q) * (1.0 - Phi_z) + std_demand * phi_z
    
    # Expected Sales = mean_demand - expected_lost_sales
    expected_sales = mean_demand - expected_lost_sales
    
    # Expected Revenue and Profit
    expected_revenue = selling_price * expected_sales + salvage_value * expected_leftover
    expected_purchase_cost = unit_cost * optimal_q
    expected_penalty_cost = stockout_penalty * expected_lost_sales
    expected_profit = expected_revenue - expected_purchase_cost - expected_penalty_cost

    return {
        "underage_cost_cu": cu,
        "overage_cost_co": co,
        "critical_ratio": critical_ratio,
        "z_star": z_star,
        "optimal_order_quantity": optimal_q,
        "optimal_quantity_rounded": math.ceil(optimal_q),
        "expected_leftover": max(0.0, expected_leftover),
        "expected_lost_sales": max(0.0, expected_lost_sales),
        "expected_sales": expected_sales,
        "expected_profit": expected_profit,
    }
