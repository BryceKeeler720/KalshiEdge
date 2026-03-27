#!/usr/bin/env bash
# Launch KalshiEdge agent + dashboard together
set -e

cd "$(dirname "$0")"
source .venv/bin/activate

echo "Starting KalshiEdge dashboard on :8050..."
uvicorn kalshiedge.dashboard:app --host 0.0.0.0 --port 8050 &
DASH_PID=$!

echo "Starting KalshiEdge agent loop..."
python -m kalshiedge.main &
AGENT_PID=$!

trap "echo 'Shutting down...'; kill $DASH_PID $AGENT_PID 2>/dev/null; wait" SIGTERM SIGINT

echo "KalshiEdge running (agent=$AGENT_PID, dashboard=$DASH_PID)"
echo "Dashboard: http://localhost:8050"
wait
