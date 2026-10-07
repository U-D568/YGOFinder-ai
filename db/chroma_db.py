import os
from typing import List

import chromadb
import dotenv


class ChromaDBConnection:
    def __init__(self):
        dotenv.load_dotenv()
        self.host = os.environ.get("host")
        self.port = os.environ.get("chroma_port")
        self.collection = os.environ.get("chroma_collection")

        self.client = chromadb.HttpClient(self.host, self.port)
        self.collection = self.client.get_or_create_collection(
            name=self.chroma_collection, metadata={"hnsw:space": "cosine"}
        )

    def remove_none_values(self, metadata: List[dict]):
        return [self.remove_none_value(data) for data in metadata]

    def remove_none_value(self, metadata: dict):
        for key in metadata:
            if metadata[key] is None:
                metadata[key] = ""
        return metadata

    def update_one(self, id, embed, metadata: dict):
        metadata = self.remove_none_value(metadata)
        return self.collection.update(
            ids=[id],
            metadatas=[metadata],
            embeddings=[embed],
        )

    def insert_one(self, id, embed, metadata: dict):
        metadata = self.remove_none_value(metadata)
        return self.collection.add(
            ids=[id],
            metadatas=[metadata],
            embeddings=[embed],
        )

    def insert(self, ids, embeds, metadata: List[dict]):
        metadata = self.remove_none_values(metadata)
        return self.collection.add(ids=ids, metadatas=metadata, embeddings=embeds)

    def search_by_embed(self, embed, n_result=1):
        result = self.collection.query(query_embeddings=[embed], n_results=n_result)
        return result["metadatas"]
