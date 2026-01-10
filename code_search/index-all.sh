#!/bin/bash
# Index all repos to PostgreSQL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

source .venv/bin/activate

DB_URL="${VPS_DB_URL:-$1}"
TRITON_URL="${TRITON_URL:-localhost:8020}"

if [[ -z "$DB_URL" ]]; then
    echo "Usage: $0 [db_url]"
    echo "Or set VPS_DB_URL in .env"
    exit 1
fi

for repo in /home/maxjeffwell/GitHub_Projects/*/; do
  name=$(basename "$repo")

  # Skip non-repos, current project, and problematic repos with node_modules
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

  # Skip if not a directory
  if [[ ! -d "$repo" ]]; then
    continue
  fi

  echo "=== Indexing: $name ==="
  python3 indexer.py "$repo" --repo-name "$name" --db-url "$DB_URL" --triton-url "$TRITON_URL" --batch-size 32
  echo ""
done

echo "Done! Checking total indexed chunks:"
python3 -c "
import psycopg2
conn = psycopg2.connect('$DB_URL')
cur = conn.cursor()
cur.execute('SELECT repo_name, COUNT(*) FROM code_embeddings GROUP BY repo_name ORDER BY COUNT(*) DESC')
for r in cur.fetchall():
    print(f'  {r[0]}: {r[1]} chunks')
cur.execute('SELECT COUNT(*) FROM code_embeddings')
print(f'Total: {cur.fetchone()[0]} chunks')
conn.close()
"
