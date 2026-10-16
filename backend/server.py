"""
server.py - ProfitGuard AI (Minimal, Light SaaS UI matching SalesPro & Tvora)
Clean off-white background, crisp white cards, natural human-generated aesthetic.
Datathon 2026 - Track 4 (Business / The Money Leak)
"""

import os
import json
import csv
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = int(os.environ.get("PROFITGUARD_PORT", "8501"))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_dataset():
    products = {}
    prod_path = os.path.join(DATA_DIR, "products.csv")
    with open(prod_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            products[r["product_id"]] = {
                "category": r["category"],
                "unit_cost": float(r["unit_cost"]),
                "list_price": float(r["list_price"]),
                "supplier_lead_days": int(r["supplier_lead_days"])
            }

    transactions = []
    cat_revenue = {}
    cat_margin = {}
    status_counts = {"Completed": 0, "Returned": 0, "Cancelled": 0, "Pending": 0}
    cust_orders = {}

    total_completed_rev = 0.0
    total_leakage_loss = 0.0
    leakage_count = 0
    profitable_count = 0

    trans_path = os.path.join(DATA_DIR, "transactions.csv")
    with open(trans_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            st = r["status"]
            status_counts[st] = status_counts.get(st, 0) + 1
            pid = r["product_id"]
            prod = products.get(pid, {"category": "General", "unit_cost": 50.0, "list_price": 100.0})

            q = float(r["quantity"])
            p = float(r["unit_price"])
            d = float(r["discount_pct"])
            eff_p = p * (1.0 - d)
            cost = prod["unit_cost"]
            unit_m = eff_p - cost
            tot_m = q * unit_m
            rev = q * eff_p

            is_completed = (st == "Completed")
            is_leakage = (is_completed and tot_m < 0)

            if is_completed:
                total_completed_rev += rev
                cat = prod["category"]
                cat_revenue[cat] = cat_revenue.get(cat, 0.0) + rev
                cat_margin[cat] = cat_margin.get(cat, 0.0) + tot_m

                if is_leakage:
                    leakage_count += 1
                    total_leakage_loss += abs(tot_m)
                else:
                    profitable_count += 1

                cid = r["customer_id"]
                dt = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
                if cid not in cust_orders:
                    cust_orders[cid] = []
                cust_orders[cid].append((dt, rev))

            # Reason & tag
            if is_leakage:
                if p < cost:
                    reason = "Price Below Cost"
                    tag = "Below Cost"
                else:
                    reason = "Excessive Promo"
                    tag = "Promo Leak"
            elif st == "Returned":
                reason = "Customer Return"
                tag = "Return"
            elif st == "Cancelled":
                reason = "Order Cancelled"
                tag = "Cancelled"
            else:
                reason = "Profitable Sale"
                tag = "Healthy"

            transactions.append({
                "id": r["transaction_id"],
                "customer_id": r["customer_id"],
                "product_id": pid,
                "category": prod["category"],
                "date": r["timestamp"][:16],
                "status": st,
                "quantity": int(q),
                "unit_price": round(p, 2),
                "discount_pct": round(d * 100, 1),
                "effective_price": round(eff_p, 2),
                "unit_cost": round(cost, 2),
                "unit_margin": round(unit_m, 2),
                "total_margin": round(tot_m, 2),
                "revenue": round(rev, 2),
                "is_leakage": is_leakage,
                "reason": reason,
                "tag": tag,
            })

    # Sample RFM
    ref_date = max(
        (max(order[0] for order in orders) for orders in cust_orders.values()),
        default=datetime.now()
    )
    rfm_list = []
    for cid, orders in cust_orders.items():
        max_dt = max(o[0] for o in orders)
        rec = (ref_date - max_dt).days
        freq = len(orders)
        mon = sum(o[1] for o in orders)
        rfm_list.append({
            "customer_id": cid,
            "recency": rec,
            "frequency": freq,
            "monetary": round(mon, 2)
        })

    return {
        "total_tx": len(transactions),
        "completed_tx": status_counts.get("Completed", 0),
        "returned_tx": status_counts.get("Returned", 0),
        "cancelled_tx": status_counts.get("Cancelled", 0),
        "profitable_tx": profitable_count,
        "leakage_tx": leakage_count,
        "total_completed_rev": round(total_completed_rev, 2),
        "total_leakage_loss": round(total_leakage_loss, 2),
        "status_counts": status_counts,
        "cat_revenue": {k: round(v, 2) for k, v in cat_revenue.items()},
        "cat_margin": {k: round(v, 2) for k, v in cat_margin.items()},
        "recent_orders": transactions,
        "worst_leaks": sorted([t for t in transactions if t["is_leakage"]], key=lambda x: x["total_margin"])[:20],
        "rfm_customers": rfm_list
    }


DATA = load_dataset()


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(DATA).encode("utf-8"))
            return

        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


HTML_PATH = os.path.join(os.path.dirname(__file__), "..", "index.html")
with open(HTML_PATH, "r", encoding="utf-8") as f:
    HTML_TEMPLATE = f.read()


def run_server():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"ProfitGuard AI Dashboard running on http://localhost:{PORT}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
