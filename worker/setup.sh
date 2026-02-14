#!/bin/bash
set -e

echo "=== LEM Worker Setup ==="
echo ""

# Check platform
if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: MLX requires Apple Silicon (M1/M2/M3/M4)."
    echo "For non-Apple hardware, use the --backend api option with llama.cpp or Ollama."
    echo ""
fi

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Install Python 3.9+."
    exit 1
fi

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYVER"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Check InfluxDB token
echo ""
if [ -f "$HOME/.influx_token" ]; then
    echo "InfluxDB token: found at ~/.influx_token"
elif [ -n "$INFLUX_TOKEN" ]; then
    echo "InfluxDB token: found in INFLUX_TOKEN env"
else
    echo "InfluxDB token: NOT FOUND"
    echo ""
    echo "  You need an InfluxDB token to coordinate with other workers."
    echo "  Get it from the team and save it:"
    echo ""
    echo "    echo 'YOUR_TOKEN_HERE' > ~/.influx_token"
    echo ""
fi

# Check InfluxDB connectivity
echo ""
INFLUX_URL="${INFLUX_URL:-http://10.69.69.165:8181}"
echo -n "InfluxDB ($INFLUX_URL): "
if python3 -c "
import urllib.request, json, os
from pathlib import Path
token = os.environ.get('INFLUX_TOKEN', '')
if not token:
    tp = Path.home() / '.influx_token'
    if tp.exists(): token = tp.read_text().strip()
if not token:
    print('SKIP (no token)')
    exit(0)
body = json.dumps({'db': 'training', 'q': 'SELECT 1 AS ok'}).encode()
req = urllib.request.Request(
    f'{os.environ.get(\"INFLUX_URL\", \"http://10.69.69.165:8181\")}/api/v3/query_sql',
    data=body, headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
urllib.request.urlopen(req, timeout=5)
print('OK')
" 2>/dev/null; then
    :
else
    echo "UNREACHABLE"
    echo "  Make sure you're on the lab network (VLAN 69) or have VPN access."
fi

# Check data files
echo ""
echo "Data files:"
for f in data/gold-prompts.jsonl data/expansion-prompts.jsonl; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        size=$(du -h "$f" | cut -f1)
        echo "  $f: $lines prompts ($size)"
    else
        echo "  $f: NOT FOUND"
    fi
done

# Summary
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick start:"
echo ""
echo "  # Gold generation (finish the 15K golden set):"
echo "  python3 lem_generate.py --worker $(hostname)-gold --dry-run"
echo "  python3 lem_generate.py --worker $(hostname)-gold"
echo ""
echo "  # Expansion generation (46K+ prompts, needs trained LEM model):"
echo "  python3 lem_expand.py --worker $(hostname)-expand --dry-run"
echo "  python3 lem_expand.py --worker $(hostname)-expand"
echo ""
echo "  # Use a smaller model for limited RAM:"
echo "  python3 lem_generate.py --model mlx-community/gemma-3-4b-it-qat-4bit"
echo ""
echo "  # Use API backend (llama.cpp, Ollama, etc.):"
echo "  python3 lem_expand.py --backend api --api-url http://localhost:8080/v1"
echo ""
