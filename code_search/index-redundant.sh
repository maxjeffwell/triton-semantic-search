#!/bin/bash
# Index repos to BOTH VPS PostgreSQL and Neon (redundancy)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

source .venv/bin/activate

VPS_URL="${VPS_DB_URL}"
NEON_URL="${NEON_DB_URL}"
TRITON_URL="${TRITON_URL:-localhost:8020}"
CF_HOSTNAME="${CLOUDFLARE_HOSTNAME}"

if [[ -z "$VPS_URL" ]] || [[ -z "$NEON_URL" ]]; then
    echo "Error: Set VPS_DB_URL and NEON_DB_URL in .env"
    exit 1
fi

# Check tunnel for VPS
if ! nc -z localhost 5433 2>/dev/null && [[ -n "$CF_HOSTNAME" ]]; then
    echo "Starting Cloudflare tunnel..."
    cloudflared access tcp --hostname "$CF_HOSTNAME" --url localhost:5433 &
    sleep 2
fi

for repo in /home/maxjeffwell/GitHub_Projects/*/; do
  name=$(basename "$repo")

  # Skip non-repos, current project, and problematic repos
  if [[ "$name" == "triton-semantic-search" ]] || \
     [[ "$name" == "spaced-repetition-capstone" ]] || \
     [[ "$name" == "github-readme-stats" ]] || \
     [[ "$name" == "bookmarks-react-hooks" ]] || \
     [[ "$name" == "educationELLy" ]] || \
     [[ "$name" == "educationELLy-graphql" ]] || \
     [[ "$name" == "ai-writing-studio" ]]; then
    echo "Skipping: $name (excluded)"
    continue
  fi

  if [[ ! -d "$repo" ]]; then
    continue
  fi

  echo "=== Indexing: $name ==="

  echo "  → VPS PostgreSQL"
  python3 indexer.py "$repo" --repo-name "$name" --db-url "$VPS_URL" --triton-url "$TRITON_URL" --batch-size 32

  echo "  → Neon"
  python3 indexer.py "$repo" --repo-name "$name" --db-url "$NEON_URL" --triton-url "$TRITON_URL" --batch-size 32

  echo ""
done

echo "=== Summary ==="
echo "VPS PostgreSQL:"
python3 -c "
import psycopg2
conn = psycopg2.connect('$VPS_URL')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM code_embeddings')
print(f'  Total: {cur.fetchone()[0]} chunks')
conn.close()
"

echo "Neon:"
python3 -c "
import psycopg2
conn = psycopg2.connect('$NEON_URL')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM code_embeddings')
print(f'  Total: {cur.fetchone()[0]} chunks')
conn.close()
"
