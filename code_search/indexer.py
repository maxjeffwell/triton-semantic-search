#!/usr/bin/env python3
"""
Code Indexer for Semantic Search

Indexes code repositories into PostgreSQL with pgvector embeddings.
Uses local Triton server for fast GPU-accelerated embedding generation.

Usage:
    python indexer.py /path/to/repo --db-url postgresql://user:pass@host:5432/db
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Generator
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import tritonclient.http as httpclient
from transformers import AutoTokenizer


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


class TritonEmbedder:
    """Generate embeddings using Triton server"""

    def __init__(self, triton_url: str = "localhost:8020"):
        self.client = httpclient.InferenceServerClient(url=triton_url)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model_name = "all-minilm-l6-v2"

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        encoded = self.tokenizer(
            texts,
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


def scan_repository(repo_path: str) -> Generator[CodeChunk, None, None]:
    """Scan a repository and yield code chunks"""
    repo_path = Path(repo_path)

    for file_path in repo_path.rglob('*'):
        # Skip directories
        if file_path.is_dir():
            continue

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
                     triton_url: str = "localhost:8020", batch_size: int = 16):
    """Index a repository into PostgreSQL"""

    print(f"Indexing repository: {repo_name}")
    print(f"Path: {repo_path}")
    print(f"Database: {db_url.split('@')[1] if '@' in db_url else db_url}")
    print()

    # Initialize embedder
    embedder = TritonEmbedder(triton_url)
    print("[OK] Connected to Triton server")

    # Connect to database
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    print("[OK] Connected to PostgreSQL")
    print()

    # Collect chunks
    chunks = list(scan_repository(repo_path))
    print(f"Found {len(chunks)} code chunks to index")

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
            """
            INSERT INTO code_embeddings
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

        conn.commit()
        indexed += len(batch)
        print(f"Indexed {indexed}/{len(chunks)} chunks", end='\r')

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

    args = parser.parse_args()

    repo_name = args.repo_name or Path(args.repo_path).name

    index_repository(
        repo_path=args.repo_path,
        repo_name=repo_name,
        db_url=args.db_url,
        triton_url=args.triton_url,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
