#!/bin/bash
# News Data Collector Setup Script
# Run this script with sudo to install the systemd service
#
# Purpose:
#   - Collects fresh news articles from Yahoo Finance (every 4 hours)
#   - Summarizes articles using abstractive methods (Lexrank, Textrank)
#   - Aligns news sequences with price data for training
#
# Note:
#   - FinBERT embedding is handled by news-embedding-batch.timer (weekly)
#   - For one-time full preprocessing, see setup_news_embedding_batch_service.sh

set -e

SERVICE_NAME="news-collector"
SERVICE_FILE="infra/news-collector.service"
PROJECT_DIR="/home/joppe-kietselaer/Desktop/coding/tradingAgent"

echo "=== News Data Collector Service Setup ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo bash setup_news_collector_service.sh"
    exit 1
fi

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"
chown joppe-kietselaer:joppe-kietselaer "$PROJECT_DIR/logs"

# Copy service file
cp "$PROJECT_DIR/$SERVICE_FILE" /etc/systemd/system/news-collector.service

# Reload systemd
systemctl daemon-reload

echo "Service installed! Here's how to use it:"
echo ""
echo "  START the collector:"
echo "    sudo systemctl start $SERVICE_NAME"
echo ""
echo "  STOP the collector:"
echo "    sudo systemctl stop $SERVICE_NAME"
echo ""
echo "  ENABLE auto-start on boot:"
echo "    sudo systemctl enable $SERVICE_NAME"
echo ""
echo "  CHECK status:"
echo "    sudo systemctl status $SERVICE_NAME"
echo ""
echo "  VIEW logs:"
echo "    tail -f $PROJECT_DIR/logs/news_collector.log"
echo ""
echo "  VIEW errors:"
echo "    tail -f $PROJECT_DIR/logs/news_collector_error.log"
echo ""
echo "📌 COMPANION SERVICE:"
echo "   For FinBERT embedding (weekly), install the batch processor:"
echo "    sudo bash setup_news_embedding_batch_service.sh"
echo ""
echo "=== Setup Complete ==="

