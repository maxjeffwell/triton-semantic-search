#!/usr/bin/env python3
"""
Code Indexer for Semantic Search

Indexes code repositories into PostgreSQL with pgvector embeddings.

Default model "e5" is embedded by the in-cluster OVMS (Intel iGPU, e5-large,
1024 d). "minilm" is the legacy 384-d index that needs the retired Triton server.

Usage:
    python indexer.py /path/to/repo --db-url postgresql://user:pass@host:5432/db
    python indexer.py /path/to/repo --db-url ... --replace   # drop this repo's rows first
"""

import os
import re
import argparse
import subprocess
from pathlib import Path
from typing import List, Generator
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from ovms_embed import OvmsEmbedder, DEFAULT_OVMS_URL


# File extensions to index by language
LANGUAGE_EXTENSIONS = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.jsx': 'javascript',
    '.go': 'go',
    '.rs': 'rust',
    '.java': 'java',
    '.cpp': 'cpp',
    '.c': 'c',
    '.h': 'c',
    '.rb': 'ruby',
    '.php': 'php',
    '.sql': 'sql',
    '.sh': 'bash',
    '.md': 'markdown',
}

# Directories to skip
SKIP_DIRS = {
    'node_modules', '.git', '__pycache__', '.venv', 'venv',
    'dist', 'build', '.next', '.cache', 'vendor', 'target'
}


class CodeChunk:
    """Represents a chunk of code to be indexed"""
    def __init__(self, file_path: str, chunk_type: str, name: str,
                 content: str, start_line: int, end_line: int, language: str):
        self.file_path = file_path
        self.chunk_type = chunk_type
        self.name = name
        self.content = content
        self.start_line = start_line
        self.end_line = end_line
        self.language = language

    def to_text(self) -> str:
        """Convert chunk to text for embedding"""
        if self.chunk_type == 'function':
            return f"{self.language} function {self.name}: {self.content[:500]}"
        elif self.chunk_type == 'class':
            return f"{self.language} class {self.name}: {self.content[:500]}"
        else:
            return f"{self.file_path}: {self.content[:500]}"


# Model configurations
MODELS = {
    'minilm': {
        'triton_name': 'all-minilm-l6-v2',
        'tokenizer': 'sentence-transformers/all-MiniLM-L6-v2',
        'dims': 384,
        'table': 'code_embeddings',
        'prefix': '',  # No prefix needed
    },
    'e5': {
        'backend': 'ovms',
        'ovms_model': 'e5-large',  # OVMS on the Intel iGPU (2026-10-01; was Triton e5-large-v2)
        'dims': 1024,
        'table': 'code_embeddings_e5',
        'prefix': 'passage: ',  # e5 uses passage prefix for documents
    }
}
MODELS['minilm']['backend'] = 'triton'  # legacy: Triton was retired 2026-09-30


class TritonEmbedder:
    """Generate embeddings using Triton server"""

    def __init__(self, triton_url: str = "localhost:8020", model: str = "minilm"):
        # Imported lazily: only the legacy minilm index needs Triton.
        import tritonclient.http as httpclient
        from transformers import AutoTokenizer
        self.httpclient = httpclient
        self.client = httpclient.InferenceServerClient(url=triton_url)
        self.config = MODELS[model]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer'])
        self.model_name = self.config['triton_name']
        self.prefix = self.config['prefix']

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        # Add prefix if model requires it (e.g., e5 uses "passage: " for documents)
        if self.prefix:
            texts = [self.prefix + t for t in texts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="np"
        )

        input_ids = self.httpclient.InferInput("input_ids", encoded["input_ids"].shape, "INT64")
        attention_mask = self.httpclient.InferInput("attention_mask", encoded["attention_mask"].shape, "INT64")
        token_type_ids = self.httpclient.InferInput("token_type_ids", encoded["input_ids"].shape, "INT64")

        input_ids.set_data_from_numpy(encoded["input_ids"].astype(np.int64))
        attention_mask.set_data_from_numpy(encoded["attention_mask"].astype(np.int64))
        token_type_ids.set_data_from_numpy(np.zeros_like(encoded["input_ids"], dtype=np.int64))

        output = self.httpclient.InferRequestedOutput("last_hidden_state")

        response = self.client.infer(
            model_name=self.model_name,
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=[output]
        )

        token_embeddings = response.as_numpy("last_hidden_state")

        # Mean pooling
        mask = encoded["attention_mask"][:, :, np.newaxis].astype(np.float32)
        embeddings = np.sum(token_embeddings * mask, axis=1) / np.sum(mask, axis=1)

        # L2 normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings


def extract_python_chunks(content: str, file_path: str) -> List[CodeChunk]:
    """Extract functions and classes from Python code"""
    chunks = []
    lines = content.split('\n')

    # Simple regex patterns for Python
    func_pattern = re.compile(r'^(\s*)def\s+(\w+)\s*\(')
    class_pattern = re.compile(r'^(\s*)class\s+(\w+)')

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for function
        func_match = func_pattern.match(line)
        if func_match:
            indent = len(func_match.group(1))
            name = func_match.group(2)
            start_line = i + 1

            # Find end of function
            j = i + 1
            while j < len(lines):
                if lines[j].strip() and not lines[j].startswith(' ' * (indent + 1)) and not lines[j].startswith('\t' * (indent // 4 + 1)):
                    if not lines[j].strip().startswith('#'):
                        break
                j += 1

            func_content = '\n'.join(lines[i:j])
            chunks.append(CodeChunk(
                file_path=file_path,
                chunk_type='function',
                name=name,
                content=func_content,
                start_line=start_line,
                end_line=j,
                language='python'
            ))
            i = j
            continue

        # Check for class
        class_match = class_pattern.match(line)
        if class_match:
            indent = len(class_match.group(1))
            name = class_match.group(2)
            start_line = i + 1

            # Find end of class
            j = i + 1
            while j < len(lines):
                if lines[j].strip() and not lines[j].startswith(' ') and not lines[j].startswith('\t'):
                    break
                j += 1

            class_content = '\n'.join(lines[i:j])
            chunks.append(CodeChunk(
                file_path=file_path,
                chunk_type='class',
                name=name,
                content=class_content,
                start_line=start_line,
                end_line=j,
                language='python'
            ))
            i = j
            continue

        i += 1

    # If no chunks found, index the whole file
    if not chunks and content.strip():
        chunks.append(CodeChunk(
            file_path=file_path,
            chunk_type='file',
            name=Path(file_path).name,
            content=content,
            start_line=1,
            end_line=len(lines),
            language='python'
        ))

    return chunks


def extract_chunks(content: str, file_path: str, language: str) -> List[CodeChunk]:
    """Extract code chunks based on language"""
    if language == 'python':
        return extract_python_chunks(content, file_path)

    # For other languages, just index the whole file for now
    lines = content.split('\n')
    return [CodeChunk(
        file_path=file_path,
        chunk_type='file',
        name=Path(file_path).name,
        content=content,
        start_line=1,
        end_line=len(lines),
        language=language
    )]


def iter_repo_files(repo_path: Path) -> Generator[Path, None, None]:
    """Tracked files via git (skips node_modules/build output, follows submodules).
    Falls back to a pruned os.walk outside git. rglob('*') used to descend into
    node_modules, which is why several repos had been excluded from indexing."""
    if (repo_path / '.git').exists():
        try:
            out = subprocess.run(
                ['git', '-C', str(repo_path), 'ls-files', '-z', '--recurse-submodules'],
                check=True, capture_output=True
            ).stdout.decode('utf-8', errors='ignore')
            for rel in filter(None, out.split('\0')):
                p = repo_path / rel
                if p.is_file():
                    yield p
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            yield Path(root) / f


def scan_repository(repo_path: str) -> Generator[CodeChunk, None, None]:
    """Scan a repository and yield code chunks"""
    repo_path = Path(repo_path)

    for file_path in iter_repo_files(repo_path):

        # Skip ignored directories
        if any(skip in file_path.parts for skip in SKIP_DIRS):
            continue

        # Check if we should index this file
        ext = file_path.suffix.lower()
        if ext not in LANGUAGE_EXTENSIONS:
            continue

        language = LANGUAGE_EXTENSIONS[ext]

        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if len(content.strip()) == 0:
                continue

            relative_path = str(file_path.relative_to(repo_path))
            chunks = extract_chunks(content, relative_path, language)

            for chunk in chunks:
                yield chunk

        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")


def index_repository(repo_path: str, repo_name: str, db_url: str,
                     triton_url: str = "localhost:8020", batch_size: int = 16,
                     model: str = "e5", ovms_url: str = DEFAULT_OVMS_URL,
                     replace: bool = False):
    """Index a repository into PostgreSQL"""

    model_config = MODELS[model]
    table_name = model_config['table']

    print(f"Indexing repository: {repo_name}")
    print(f"Path: {repo_path}")
    print(f"Model: {model} ({model_config['dims']} dims)")
    print(f"Table: {table_name}")
    print(f"Database: {db_url.split('@')[1] if '@' in db_url else db_url}")
    print()

    # Initialize embedder
    if model_config['backend'] == 'ovms':
        embedder = OvmsEmbedder(ovms_url, model=model_config['ovms_model'],
                                prefix=model_config['prefix'], dims=model_config['dims'])
        print(f"[OK] Using OVMS {ovms_url} ({model_config['ovms_model']})")
    else:
        embedder = TritonEmbedder(triton_url, model=model)
        print(f"[OK] Connected to Triton server ({model_config['triton_name']})")

    # Connect to database
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    print("[OK] Connected to PostgreSQL")
    print()

    # Collect chunks
    chunks = list(scan_repository(repo_path))
    print(f"Found {len(chunks)} code chunks to index")

    # --replace: drop this repo's rows in the SAME transaction as the inserts, so
    # deleted files/functions and vectors from an older model don't linger, and a
    # failed run rolls back to the previous index instead of leaving it half-empty.
    if replace:
        cur.execute(f"DELETE FROM {table_name} WHERE repo_name = %s", (repo_name,))
        print(f"Replacing: removed {cur.rowcount} existing rows for {repo_name}")

    # Process in batches
    indexed = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Generate embeddings
        texts = [chunk.to_text() for chunk in batch]
        embeddings = embedder.embed_batch(texts)

        # Prepare data for insertion
        rows = []
        for chunk, embedding in zip(batch, embeddings):
            rows.append((
                repo_name,
                chunk.file_path,
                chunk.chunk_type,
                chunk.name,
                chunk.content[:10000],  # Limit content size
                chunk.start_line,
                chunk.end_line,
                embedding.tolist(),
                chunk.language
            ))

        # Insert into database
        execute_values(
            cur,
            f"""
            INSERT INTO {table_name}
            (repo_name, file_path, chunk_type, name, content, start_line, end_line, embedding, language)
            VALUES %s
            ON CONFLICT (repo_name, file_path, chunk_type, name, start_line)
            DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                indexed_at = CURRENT_TIMESTAMP
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)"
        )

        if not replace:
            conn.commit()
        indexed += len(batch)
        print(f"Indexed {indexed}/{len(chunks)} chunks", end='\r')

    conn.commit()
    print(f"\nDone! Indexed {indexed} chunks from {repo_name}")

    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Index code repository for semantic search")
    parser.add_argument("repo_path", help="Path to repository")
    parser.add_argument("--repo-name", help="Name for the repository (default: directory name)")
    parser.add_argument("--db-url", required=True, help="PostgreSQL connection URL")
    parser.add_argument("--triton-url", default="localhost:8020", help="Triton server URL")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding")
    parser.add_argument("--model", choices=['minilm', 'e5'], default="e5",
                        help="Embedding model: e5 (1024d, OVMS, default) or minilm (384d, legacy Triton)")
    parser.add_argument("--ovms-url", default=DEFAULT_OVMS_URL, help="Embedding endpoint: gateway /api/ai/embed (default) or an OVMS base URL (env EMBED_URL)")
    parser.add_argument("--replace", action="store_true",
                        help="Delete this repo's existing rows first (same transaction)")

    args = parser.parse_args()

    repo_name = args.repo_name or Path(args.repo_path).name

    index_repository(
        repo_path=args.repo_path,
        repo_name=repo_name,
        db_url=args.db_url,
        triton_url=args.triton_url,
        batch_size=args.batch_size,
        model=args.model,
        ovms_url=args.ovms_url,
        replace=args.replace
    )


if __name__ == "__main__":
    main()
