#!/bin/bash
# News FinBERT Batch Embedding Service Setup Script
# Run this script with sudo to install the systemd service and timer
#
# Purpose:
#   - Automatically encodes all news articles to FinBERT embeddings weekly
#   - Runs every Sunday at 2:00 AM (off-peak) for ~2-3 hours
#   - Maintains full coverage of all 6,357 tickers with news data
#   - Integrates with minute-collector and news-collector services
#
# Usage:
#   sudo bash setup_news_embedding_batch_service.sh
#
# After installation:
#   - Check timer: systemctl list-timers news-embedding-batch.timer
#   - View logs: tail -f logs/news_embedding.log
#   - Manually trigger: sudo systemctl start news-embedding-batch.service

set -e

SERVICE_NAME="news-embedding-batch"
SERVICE_FILE="infra/news-embedding-batch.service"
TIMER_FILE="infra/news-embedding-batch.timer"
PROJECT_DIR="/home/joppe-kietselaer/Desktop/coding/tradingAgent"

echo "=== News FinBERT Batch Embedding Service Setup ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo bash setup_news_embedding_batch_service.sh"
    exit 1
fi

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"
chown joppe-kietselaer:joppe-kietselaer "$PROJECT_DIR/logs"

# Copy service and timer files
echo "Installing service file: $SERVICE_FILE"
cp "$PROJECT_DIR/$SERVICE_FILE" /etc/systemd/system/$SERVICE_NAME.service
chmod 644 /etc/systemd/system/$SERVICE_NAME.service

echo "Installing timer file: $TIMER_FILE"
cp "$PROJECT_DIR/$TIMER_FILE" /etc/systemd/system/$SERVICE_NAME.timer
chmod 644 /etc/systemd/system/$SERVICE_NAME.timer

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

echo ""
echo "=== Service Installation Complete ==="
echo ""
echo "📋 SYSTEMD SERVICE OVERVIEW:"
echo "  Service: $SERVICE_NAME.service"
echo "  Timer:   $SERVICE_NAME.timer"
echo "  Schedule: Every Sunday at 2:00 AM"
echo "  Runtime: ~2-3 hours (FinBERT encoding of 6,357 tickers)"
echo "  GPU: CUDA (RTX 5080, batch_size=128)"
echo ""
echo "📝 HOW TO USE:"
echo ""
echo "  1️⃣  ENABLE auto-start on boot:"
echo "      sudo systemctl enable $SERVICE_NAME.timer"
echo ""
echo "  2️⃣  START the timer:"
echo "      sudo systemctl start $SERVICE_NAME.timer"
echo ""
echo "  3️⃣  CHECK timer status:"
echo "      systemctl status $SERVICE_NAME.timer"
echo ""
echo "  4️⃣  LIST all active timers (see next run time):"
echo "      systemctl list-timers $SERVICE_NAME.timer"
echo ""
echo "  5️⃣  VIEW processing logs (real-time):"
echo "      tail -f $PROJECT_DIR/logs/news_embedding.log"
echo ""
echo "  6️⃣  VIEW error logs:"
echo "      tail -f $PROJECT_DIR/logs/news_embedding_error.log"
echo ""
echo "  7️⃣  MANUALLY TRIGGER a run NOW (useful for testing/on-demand):"
echo "      sudo systemctl start $SERVICE_NAME.service"
echo ""
echo "  8️⃣  STOP the timer (disables automatic scheduling):"
echo "      sudo systemctl stop $SERVICE_NAME.timer"
echo ""
echo "  9️⃣  DISABLE timer on boot:"
echo "      sudo systemctl disable $SERVICE_NAME.timer"
echo ""
echo "🔗 INTEGRATION WITH EXISTING SERVICES:"
echo ""
echo "  ✅ minute-collector.service (every 1 hour)"
echo "     └─ Collects fresh minute data for 4,370 tickers"
echo ""
echo "  ✅ news-collector.service (every 4 hours)"
echo "     └─ Collects + summarizes fresh news articles"
echo ""
echo "  ✅ news-embedding-batch.timer (every Sunday 2 AM) [NEW]"
echo "     └─ Encodes all news articles to FinBERT embeddings"
echo ""
echo "📊 EXPECTED OUTPUT:"
echo "  - 6,357 tickers with news_articles.csv"
echo "  - ~300-500K news articles encoded"
echo "  - Storage: 6,357 × (768×4 + 6×4 + 4) bytes ≈ 20 GB in feature_cache/news/"
echo "  - Integration: Ready for --use-news training flag"
echo ""
echo "⏰ NEXT RUN TIME:"
systemctl list-timers --no-pager $SERVICE_NAME.timer || echo "  (Timer not yet started; enable it first)"
echo ""
echo "=== Setup Complete ==="
echo ""

