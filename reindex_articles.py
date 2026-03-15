#!/usr/bin/env python3
"""Re-index codigo_transito.txt using article-aware chunking."""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os, re
from dotenv import load_dotenv
load_dotenv()
import chromadb
from openai import OpenAI

COLLECTION_NAME = "transito_colombia_v2"
EMBEDDING_MODEL = "text-embedding-3-small"
PERSIST_DIR = "./chroma_db"

def split_by_articles(text):
    pattern = r'(?=ARTÍCULO\s*\n?\s*\d+[A-Za-z-]*[\.\s])'
    parts = re.split(pattern, text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        art_match = re.match(r'ARTÍCULO\s*\n?\s*(\d+[A-Za-z-]*)[\.\s]', part)
        if art_match:
            chunks.append({'text': part, 'article': f'Artículo {art_match.group(1)}', 'source': 'codigo_transito'})
        elif len(part) > 100:
            chunks.append({'text': part, 'article': '', 'source': 'codigo_transito'})
    return chunks

def get_embeddings_batch(client, texts):
    all_emb = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[i:i+100])
        all_emb.extend([x.embedding for x in resp.data])
    return all_emb

def main():
    oc = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    cc = chromadb.PersistentClient(path=PERSIST_DIR)
    col = cc.get_collection(COLLECTION_NAME)
    print(f"Current chunks: {col.count()}")

    print("Removing old codigo_transito chunks...")
    existing = col.get(where={"source": "codigo_transito"}, include=[])
    if existing['ids']:
        col.delete(ids=existing['ids'])
        print(f"Deleted {len(existing['ids'])} chunks")

    print("Splitting by articles...")
    with open('codigo_transito.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = split_by_articles(text)
    art_chunks = [c for c in chunks if c['article']]
    print(f"Total chunks: {len(chunks)} ({len(art_chunks)} articles)")

    for i in range(0, len(chunks), 50):
        batch = chunks[i:i+50]
        print(f"Embedding batch {i//50+1}/{(len(chunks)-1)//50+1}...")
        embs = get_embeddings_batch(oc, [c['text'] for c in batch])
        col.add(
            ids=[f"ct_art_{i+j}" for j in range(len(batch))],
            embeddings=embs,
            documents=[c['text'] for c in batch],
            metadatas=[{'source': c['source'], 'article': c['article'], 'chunk_index': i+j} for j, c in enumerate(batch)]
        )

    print(f"\nDone! Collection now has {col.count()} chunks.")

    print("\nTest: 'articulo 127'...")
    q = oc.embeddings.create(model=EMBEDDING_MODEL, input="Que dice el articulo 127 de la ley 769 de 2002").data[0].embedding
    r = col.query(query_embeddings=[q], n_results=3, include=['documents','metadatas','distances'])
    for doc, meta, dist in zip(r['documents'][0], r['metadatas'][0], r['distances'][0]):
        print(f"  [{dist:.3f}] {meta.get('article','?')}: {doc[:120]}...")

    print("\nTest: 'articulo 104'...")
    q2 = oc.embeddings.create(model=EMBEDDING_MODEL, input="Que dice el articulo 104 de la ley 769").data[0].embedding
    r2 = col.query(query_embeddings=[q2], n_results=2, include=['documents','metadatas','distances'])
    for doc, meta, dist in zip(r2['documents'][0], r2['metadatas'][0], r2['distances'][0]):
        print(f"  [{dist:.3f}] {meta.get('article','?')}: {doc[:120]}...")

if __name__ == "__main__":
    main()
