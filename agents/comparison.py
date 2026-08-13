# ─────────────────────────────────────────
# AGENT 3 — Comparison Agent
# Job: Compare current scan with previous scan
# ─────────────────────────────────────────
def comparison_agent(current_scan, previous_scan):
    print(f"\n[Agent 3] 📊 Comparing current scan with previous scan...")

    def calculate_change(current_val, previous_val):
        """Calculate absolute change, percent change, and trend"""
        absolute = round(current_val - previous_val, 2)

        if previous_val == 0:
            trend = "NEW"
            percent = 0.0
        elif absolute > 0:
            trend = "GREW"
            percent = round((absolute / previous_val) * 100, 1)
        elif absolute < 0:
            trend = "SHRANK"
            percent = round((absolute / previous_val) * 100, 1)
        else:
            trend = "STABLE"
            percent = 0.0

        return {
            "absolute": absolute,
            "percent":  percent,
            "trend":    trend,
        }

    # Compare all 4 volume metrics
    total_change = calculate_change(
        current_scan["total_volume"],
        previous_scan["total_volume"]
    )
    ncr_change = calculate_change(
        current_scan["ncr_volume"],
        previous_scan["ncr_volume"]
    )
    ed_change = calculate_change(
        current_scan["ed_volume"],
        previous_scan["ed_volume"]
    )
    et_change = calculate_change(
        current_scan["et_volume"],
        previous_scan["et_volume"]
    )

    print(f"[Agent 3] Total volume change: {total_change['absolute']} cm³ ({total_change['trend']})")
    print(f"[Agent 3] ✅ Comparison complete!")

    return {
        "previous_date":      previous_scan.get("scan_date", "Unknown"),
        "prev_total_volume":  previous_scan["total_volume"],
        "prev_ncr_volume":    previous_scan["ncr_volume"],
        "prev_ed_volume":     previous_scan["ed_volume"],
        "prev_et_volume":     previous_scan["et_volume"],
        "total_change":       total_change,
        "ncr_change":         ncr_change,
        "ed_change":          ed_change,
        "et_change":          et_change,
    }
