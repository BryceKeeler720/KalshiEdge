#!/usr/bin/env bash
# Install systemd services for KalshiEdge on the LXC
set -e

cp deploy/kalshiedge.service /etc/systemd/system/
cp deploy/kalshiedge-dashboard.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable kalshiedge kalshiedge-dashboard
systemctl start kalshiedge kalshiedge-dashboard

echo "Services installed and started."
echo "  Agent:     systemctl status kalshiedge"
echo "  Dashboard: systemctl status kalshiedge-dashboard"
echo "  Logs:      journalctl -u kalshiedge -f"
