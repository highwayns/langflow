from typing import Optional, Union
from langflow import CustomComponent

from langchain.graphs.neo4j_graph import Neo4jGraph
from langchain.schema import Document
from langchain.graphs.base import GraphStore
from langchain.schema import BaseRetriever
from langchain.embeddings.base import Embeddings


class Neo4jGraphComponent(CustomComponent):
    display_name: str = "Neo4jGraph"
    description: str = "Implementation of Graph Store using Neo4jGraph"
    documentation = (
        "https://python.langchain.com/docs/integrations/graphs/Neo4jGraph"
    )
    beta = True
    # api key should be password = True
    field_config = {
        "neo4j_customer_id": {"display_name": "Neo4j Customer ID"},
        "neo4j_corpus_id": {"display_name": "Neo4j Corpus ID"},
        "neo4j_api_key": {"display_name": "Neo4j API Key", "password": True},
        "code": {"show": False},
        "documents": {"display_name": "Documents"},
        "embedding": {"display_name": "Embedding"},
    }

    def build(
        self,
        neo4j_customer_id: str,
        neo4j_corpus_id: str,
        neo4j_api_key: str,
        embedding: Optional[Embeddings] = None,
        documents: Optional[Document] = None,
    ) -> Union[GraphStore, BaseRetriever]:
        # If documents, then we need to create a Neo4j instance using .from_documents
        if documents is not None and embedding is not None:
            return Neo4j.from_documents(
                documents=documents,  # type: ignore
                neo4j_customer_id=neo4j_customer_id,
                neo4j_corpus_id=neo4j_corpus_id,
                neo4j_api_key=neo4j_api_key,
                embedding=embedding,
            )

        return Neo4jGraph(
            neo4j_customer_id=neo4j_customer_id,
            neo4j_corpus_id=neo4j_corpus_id,
            neo4j_api_key=neo4j_api_key,
        )
