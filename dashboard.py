from flask import Flask, render_template_string
import pandas as pd
import os

LOG_FILE = "trades_log.csv"
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html>
<head><title>Trading Bot Dashboard</title><meta http-equiv="refresh" content="10">
<style>body { background:#111; color:#eee; font-family:Arial;} table { width:100%; border-collapse:collapse; } th, td { padding:8px; border:1px solid #444; } th { background:#222; }</style></head>
<body>
<h1>Trading Bot Dashboard</h1>
<p>Last update: {{ last_update }}</p>
<p>Total trades: {{ total }} | Paper mode: {{ paper }} | Daily loss: {{ daily_loss }}</p>
<table><tr><th>Time</th><th>Exchange</th><th>Market</th><th>Symbol</th><th>Side</th><th>Price</th><th>Amount</th><th>Status</th><th>TP</th><th>SL</th></tr>
{% for t in trades %}
<tr>
<td>{{ t.timestamp }}</td><td>{{ t.exchange }}</td><td>{{ t.market_type }}</td><td>{{ t.symbol }}</td>
<td>{{ t.side }}</td><td>{{ t.price }}</td><td>{{ t.amount }}</td><td>{{ t.status }}</td>
<td>{{ t.get('tp','') }}</td><td>{{ t.get('sl','') }}</td>
</tr>
{% endfor %}
</table></body>
</html>
"""

@app.route('/')
def index():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        last = df['timestamp'].max()
        total = len(df)
        today = pd.Timestamp.now().date()
        daily = df[df['timestamp'].dt.date == today]
        loss = 0
        for _, r in daily.iterrows():
            if r['side'].lower() == 'sell' and r.get('status','') != 'paper':
                loss += r['amount'] * r['price']
        return TEMPLATE.format(trades=df.to_dict('records'),
                               last_update=last,
                               total=total,
                               paper=str(os.getenv("PAPER_MODE")),
                               daily_loss=loss), 200
    else:
        return TEMPLATE.format(trades=[], last_update="N/A", total=0, paper=str(os.getenv("PAPER_MODE")), daily_loss=0), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
