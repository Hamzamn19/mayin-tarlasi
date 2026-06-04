#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ "$1" = "stop" ]; then
  fuser -k 8000/tcp 2>/dev/null || true
  fuser -k 3000/tcp 2>/dev/null || true
  echo "Servers stopped."
  exit 0
fi

# Kill existing processes on ports 8000 and 3000
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 3000/tcp 2>/dev/null || true
sleep 1

echo "Starting backend..."
cd "$ROOT/backend"
nohup /usr/bin/python3.12 main.py > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID (http://localhost:8000)"

echo "Starting frontend..."
cd "$ROOT/nextjs-app"
nohup ./node_modules/.bin/next dev -p 3000 > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID (http://localhost:3000)"

echo ""
echo "Waiting for servers to start..."
sleep 5

# Verify backend
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
  echo "  Backend: OK"
else
  echo "  Backend: still starting (check /tmp/backend.log)"
fi

# Verify frontend
if curl -sf -o /dev/null http://localhost:3000; then
  echo "  Frontend: OK"
else
  echo "  Frontend: still starting (check /tmp/frontend.log)"
fi

echo ""
echo "Done. Backend at http://localhost:8000 | Frontend at http://localhost:3000"
