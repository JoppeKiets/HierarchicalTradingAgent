#!/usr/bin/env bash
# setup_paper_trader.sh — install the paper-trader systemd service and timer.
#
# Usage:
#   chmod +x setup_paper_trader.sh
#   ./setup_paper_trader.sh
#
# What it does:
#   1. Creates required data directories.
#   2. Copies the service + timer units to ~/.config/systemd/user/ (user-level,
#      no sudo needed).  Pass --system to install system-wide instead.
#   3. Enables and starts the timer.
#   4. Runs a --dry-run to validate the full pipeline before the first live run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The project root is one level up from infra/
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_PYTHON="/home/joppe-kietselaer/miniconda3/envs/trading311/bin/python"
CONDA_BIN="/home/joppe-kietselaer/miniconda3/envs/trading311/bin"
SYSTEMD_USER="$HOME/.config/systemd/user"
USE_SYSTEM=0

for arg in "$@"; do
  [[ "$arg" == "--system" ]] && USE_SYSTEM=1
done

echo "═══════════════════════════════════════════════════════════"
echo "  Paper Trader — Setup"
echo "═══════════════════════════════════════════════════════════"

# ── 1. Create data directories ────────────────────────────────────────────
echo ""
echo "▶  Creating data directories…"
mkdir -p \
  "$PROJECT_DIR/data/paper_trading/price_cache" \
  "$PROJECT_DIR/data/paper_trading/daily_snapshots" \
  "$PROJECT_DIR/logs"
echo "   ✓ data/paper_trading/"

# ── 2. Install systemd units ──────────────────────────────────────────────
echo ""
echo "▶  Installing systemd units…"

if [[ $USE_SYSTEM -eq 1 ]]; then
  UNIT_DIR="/etc/systemd/system"
  SYSTEMCTL="sudo systemctl"
  echo "   Installing system-wide to $UNIT_DIR (needs sudo)"
else
  UNIT_DIR="$SYSTEMD_USER"
  SYSTEMCTL="systemctl --user"
  mkdir -p "$UNIT_DIR"
  echo "   Installing user-level to $UNIT_DIR (no sudo needed)"
fi

cp "$SCRIPT_DIR/paper-trader.service"          "$UNIT_DIR/paper-trader.service"
cp "$SCRIPT_DIR/paper-trader.timer"            "$UNIT_DIR/paper-trader.timer"
cp "$SCRIPT_DIR/paper-trader-intraday.service" "$UNIT_DIR/paper-trader-intraday.service"
cp "$SCRIPT_DIR/paper-trader-intraday.timer"   "$UNIT_DIR/paper-trader-intraday.timer"
echo "   ✓ Copied paper-trader.service / .timer and paper-trader-intraday.service / .timer"

# ── 3. Patch the WorkingDirectory and ExecStart in the installed unit ─────
# (in case the project lives in a different path than the default)
sed -i "s|WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|" "$UNIT_DIR/paper-trader.service"
sed -i "s|WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|" "$UNIT_DIR/paper-trader-intraday.service"
sed -i "s|Environment=\"PATH=.*|Environment=\"PATH=$CONDA_BIN:/usr/local/bin:/usr/bin:/bin\""|" \
       "$UNIT_DIR/paper-trader.service" "$UNIT_DIR/paper-trader-intraday.service"
sed -i "s|ExecStart=.*/bin/python|ExecStart=$CONDA_PYTHON|g" \
       "$UNIT_DIR/paper-trader.service" "$UNIT_DIR/paper-trader-intraday.service"
sed -i "s|/home/.*/tradingAgent/scripts/paper_trader.py|$PROJECT_DIR/scripts/paper_trader.py|g" \
       "$UNIT_DIR/paper-trader.service" "$UNIT_DIR/paper-trader-intraday.service"
echo "   ✓ Patched paths in service units"

# ── 4. Reload, enable and start the timer ────────────────────────────────
echo ""
echo "▶  Enabling and starting timer…"
$SYSTEMCTL daemon-reload
$SYSTEMCTL enable paper-trader.timer
$SYSTEMCTL start  paper-trader.timer
$SYSTEMCTL enable paper-trader-intraday.timer
$SYSTEMCTL start  paper-trader-intraday.timer
echo "   ✓ Both timers active"
$SYSTEMCTL status paper-trader.timer paper-trader-intraday.timer --no-pager -l || true

# ── 5. Dry-run sanity check ───────────────────────────────────────────────
echo ""
echo "▶  Running dry-run sanity check (no data written)…"
cd "$PROJECT_DIR"
"$CONDA_PYTHON" scripts/paper_trader.py --dry-run 2>&1 | tail -30
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
  echo "  Timer fires Mon–Fri at 14:35 UTC (09:35 ET) for daily signals."
  echo "  Intraday timer checks stops/TP every 15 min during market hours."
  echo "  Check status:    $SYSTEMCTL status paper-trader.timer paper-trader-intraday.timer"
  echo "  View logs:       journalctl --user -u paper-trader.service -f"
  echo "  Intraday logs:   journalctl --user -u paper-trader-intraday.service -f"
  echo "  Run manually:    $SYSTEMCTL start paper-trader.service"
  echo "  Dashboard:       python scripts/trading_dashboard.py"
  echo "  Watch live:      python scripts/trading_dashboard.py --watch"
echo "═══════════════════════════════════════════════════════════"
