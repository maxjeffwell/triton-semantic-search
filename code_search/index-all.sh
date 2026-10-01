#!/bin/bash
# Re-index every repo under ~/GitHub_Projects into the e5 code index
# (table code_embeddings_e5 on cnpg-bookmarked via code-search-db.el-jefe.me,
# embeddings from OVMS e5-large via the AI gateway).
# Each repo is replaced atomically (--replace): stale chunks and vectors from an
# older model are dropped in the same transaction as the new inserts.
#
# Usage: ./index-all.sh [db_url]     (or CODE_SEARCH_DB_URL in .env)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi
source .venv/bin/activate

DB_URL="${1:-$CODE_SEARCH_DB_URL}"
if [[ -z "$DB_URL" ]]; then
    echo "Usage: $0 [db_url]   (or set CODE_SEARCH_DB_URL in .env)"
    exit 1
fi

# Only this project and the spaced-repetition app (no indexable sources) are
# skipped. The indexer lists git-tracked files, so node_modules is never walked.
for repo in /home/maxjeffwell/GitHub_Projects/*/; do
  name=$(basename "$repo")
  if [[ "$name" == "triton-semantic-search" ]] || [[ "$name" == "spaced-repetition-capstone" ]]; then
    echo "Skipping: $name (excluded)"
    continue
  fi
  [[ -d "$repo/.git" ]] || continue

  echo "=== Indexing: $name ==="
  python3 indexer.py "$repo" --repo-name "$name" --db-url "$DB_URL" --model e5 --replace --batch-size 32 \
    | tr '\r' '\n' | grep -vE '^Indexed [0-9]+/'
  echo ""
done

echo "Done! e5 index per repo:"
python3 - "$DB_URL" <<'PY'
import sys, psycopg2
conn = psycopg2.connect(sys.argv[1]); cur = conn.cursor()
cur.execute('SELECT repo_name, COUNT(*), MAX(indexed_at)::date FROM code_embeddings_e5 GROUP BY 1 ORDER BY 2 DESC')
for r in cur.fetchall():
    print(f'  {r[0]}: {r[1]} chunks (indexed {r[2]})')
cur.execute('SELECT COUNT(*) FROM code_embeddings_e5')
print(f'Total: {cur.fetchone()[0]} chunks')
conn.close()
PY
