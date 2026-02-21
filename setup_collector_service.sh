#!/bin/bash
# Minute Data Collector Setup Script
# Run this script with sudo to install the systemd service

set -e

SERVICE_NAME="minute-collector"
SERVICE_FILE="minute-collector.service"
PROJECT_DIR="/home/joppe-kietselaer/Desktop/coding/tradingAgent"

echo "=== Minute Data Collector Service Setup ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo bash setup_collector_service.sh"
    exit 1
fi

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"
chown joppe-kietselaer:joppe-kietselaer "$PROJECT_DIR/logs"

# Copy service file
cp "$PROJECT_DIR/$SERVICE_FILE" /etc/systemd/system/

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
echo "    tail -f $PROJECT_DIR/logs/minute_collector.log"
echo ""
echo "  VIEW errors:"
echo "    tail -f $PROJECT_DIR/logs/minute_collector_error.log"
echo ""
echo "=== Setup Complete ==="
