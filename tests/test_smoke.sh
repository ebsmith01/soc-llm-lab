#!/bin/bash
set -e

echo "Running smoke test..."

# Wait for container to start
sleep 2

# Test /health
HEALTH=$(curl -s http://127.0.0.1:8000/health | jq -r '.status')

if [[ "$HEALTH" != "ok" ]]; then
  echo "❌ API did NOT return healthy!"
  exit 1
fi

echo "✔ /health OK"

# Test basic /ask
ANSWER=$(curl -s -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is access control?", "top_k": 4}' | jq -r '.answer')

if [[ -z "$ANSWER" ]]; then
  echo "❌ /ask did not return a valid answer"
  exit 1
fi

echo "✔ /ask returned a response"

echo "🎉 Smoke test PASSED!"
