#!/usr/bin/env bash
# Hermes Model Router — Demo Script
# Runs 10 sample tasks through the router and prints a summary table.
set -e

HERMES_HOME="${HERMES_HOME:-$(python3 -c 'import os; print(os.path.expanduser("~/AppData/Local/hermes"))')}"
cd "$HERMES_HOME"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                 Hermes Model Router — Live Demo                              ║"
echo "║        Classifies tasks → picks best free model → outputs switch command    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo

printf "%-4s %-45s %-15s %-12s %-40s\n" "#" "TASK" "TYPE" "PROVIDER" "MODEL"
printf "%-4s %-45s %-15s %-12s %-40s\n" "---------------------------------------------" "-----------------------------------------" "---------------" "------------" "----------------------------------------"

tasks=(
  "write a python function that merges two sorted lists"
  "analyze pros and cons of microservices vs monolith"
  "write a 4-line poem about rain in Dhaka"
  "summarize this article in one sentence"
  "hey what's up"
  "fix this bug in my python code"
  "plan a 3-day trip to Singapore"
  "debug my React component that's not rendering"
  "brainstorm 10 AI startup ideas for 2026"
  "translate hello to French and German"
)

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  result=$(python3 routing-router.py "$task" --json 2>/dev/null)
  type=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['task_type'])")
  provider=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['chosen_provider'])")
  model=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['chosen_model'])")
  switch=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('switch_command',''))")
  
  printf "%-4s %-45s %-15s %-12s %-40s\n" "$((i+1))" "$task" "$type" "$provider" "$model"
done

echo
echo "✅ Demo complete. All tasks routed to free-tier models."
