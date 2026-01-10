#!/usr/bin/env python3
"""
Code Semantic Search Client

Search your indexed code repositories using natural language.

Usage:
    python search.py "how does authentication work" --db-url postgresql://...
    python search.py "database connection" --repo my-repo --limit 5
"""

import argparse
import numpy as np
import psycopg2
import tritonclient.http as httpclient
from transformers import AutoTokenizer


# Model configurations
MODELS = {
    'minilm': {
        'triton_name': 'all-minilm-l6-v2',
        'tokenizer': 'sentence-transformers/all-MiniLM-L6-v2',
        'dims': 384,
        'table': 'code_embeddings',
        'query_prefix': '',  # No prefix needed
    },
    'e5': {
        'triton_name': 'e5-large-v2',
        'tokenizer': 'intfloat/e5-large-v2',
        'dims': 1024,
        'table': 'code_embeddings_e5',
        'query_prefix': 'query: ',  # e5 uses query prefix for search
    }
}


class CodeSearch:
    """Semantic search over indexed code"""

    def __init__(self, db_url: str, triton_url: str = "localhost:8020", model: str = "minilm"):
        self.conn = psycopg2.connect(db_url)
        self.client = httpclient.InferenceServerClient(url=triton_url)
        self.config = MODELS[model]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer'])
        self.model_name = self.config['triton_name']
        self.table_name = self.config['table']
        self.query_prefix = self.config['query_prefix']

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for query"""
        # Add prefix if model requires it (e.g., e5 uses "query: " for search)
        if self.query_prefix:
            text = self.query_prefix + text

        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="np"
        )

        input_ids = httpclient.InferInput("input_ids", encoded["input_ids"].shape, "INT64")
        attention_mask = httpclient.InferInput("attention_mask", encoded["attention_mask"].shape, "INT64")
        token_type_ids = httpclient.InferInput("token_type_ids", encoded["input_ids"].shape, "INT64")

        input_ids.set_data_from_numpy(encoded["input_ids"].astype(np.int64))
        attention_mask.set_data_from_numpy(encoded["attention_mask"].astype(np.int64))
        token_type_ids.set_data_from_numpy(np.zeros_like(encoded["input_ids"], dtype=np.int64))

        output = httpclient.InferRequestedOutput("last_hidden_state")

        response = self.client.infer(
            model_name=self.model_name,
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=[output]
        )

        token_embeddings = response.as_numpy("last_hidden_state")
        mask = encoded["attention_mask"][:, :, np.newaxis].astype(np.float32)
        embedding = np.sum(token_embeddings * mask, axis=1) / np.sum(mask, axis=1)
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        return embedding[0]

    def search(self, query: str, limit: int = 10, repo_name: str = None,
               language: str = None, chunk_type: str = None) -> list:
        """
        Search for code similar to the query.

        Args:
            query: Natural language search query
            limit: Maximum results to return
            repo_name: Filter by repository name
            language: Filter by programming language
            chunk_type: Filter by chunk type (function, class, file)

        Returns:
            List of matching code chunks with similarity scores
        """
        # Generate query embedding
        embedding = self.embed(query)
        embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'

        # Build query with filters
        filters = []
        params = []

        if repo_name:
            filters.append("repo_name = %s")
            params.append(repo_name)
        if language:
            filters.append("language = %s")
            params.append(language)
        if chunk_type:
            filters.append("chunk_type = %s")
            params.append(chunk_type)

        where_clause = ""
        if filters:
            where_clause = "WHERE " + " AND ".join(filters)

        sql = f"""
            SELECT
                repo_name,
                file_path,
                chunk_type,
                name,
                content,
                start_line,
                end_line,
                language,
                1 - (embedding <=> %s::vector) AS similarity
            FROM {self.table_name}
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        params = [embedding_str] + params + [embedding_str, limit]

        cur = self.conn.cursor()
        cur.execute(sql, params)
        results = cur.fetchall()
        cur.close()

        return [
            {
                'repo_name': r[0],
                'file_path': r[1],
                'chunk_type': r[2],
                'name': r[3],
                'content': r[4],
                'start_line': r[5],
                'end_line': r[6],
                'language': r[7],
                'similarity': float(r[8])
            }
            for r in results
        ]

    def close(self):
        self.conn.close()


def format_result(result: dict, show_content: bool = True) -> str:
    """Format a search result for display"""
    lines = []

    # Header
    sim = result['similarity']
    sim_bar = '█' * int(sim * 20) + '░' * (20 - int(sim * 20))
    lines.append(f"[{sim:.3f}] {sim_bar}")

    # Location
    loc = f"{result['repo_name']}/{result['file_path']}"
    if result['name']:
        loc += f" :: {result['chunk_type']} {result['name']}"
    if result['start_line']:
        loc += f" (L{result['start_line']}-{result['end_line']})"
    lines.append(f"  📁 {loc}")

    # Content preview
    if show_content:
        content = result['content']
        # Truncate long content
        if len(content) > 300:
            content = content[:300] + "..."
        # Indent content
        for line in content.split('\n')[:8]:
            lines.append(f"  │ {line}")
        if content.count('\n') > 8:
            lines.append(f"  │ ... ({content.count(chr(10)) - 8} more lines)")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Search code with natural language")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--db-url", required=True, help="PostgreSQL connection URL")
    parser.add_argument("--triton-url", default="localhost:8020", help="Triton server URL")
    parser.add_argument("--repo", help="Filter by repository name")
    parser.add_argument("--language", help="Filter by language (python, javascript, etc.)")
    parser.add_argument("--type", choices=['function', 'class', 'file'], help="Filter by chunk type")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--no-content", action="store_true", help="Don't show code content")
    parser.add_argument("--model", choices=['minilm', 'e5'], default="minilm",
                        help="Embedding model: minilm (384d, fast) or e5 (1024d, quality)")

    args = parser.parse_args()

    model_info = MODELS[args.model]
    print(f"🔍 Searching: \"{args.query}\" [{args.model}]")
    print()

    search = CodeSearch(args.db_url, args.triton_url, model=args.model)

    results = search.search(
        query=args.query,
        limit=args.limit,
        repo_name=args.repo,
        language=args.language,
        chunk_type=args.type
    )

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"{'─' * 60}")
        print(f"Result {i}")
        print(format_result(result, show_content=not args.no_content))
        print()

    search.close()


if __name__ == "__main__":
    main()
