from typing import Optional, Union
from langflow import CustomComponent

from langchain.graphs.hugegraph import HugeGraph
from langchain.schema import Document
from langchain.graphs.base import GraphStore
from langchain.schema import BaseRetriever
from langchain.embeddings.base import Embeddings


class HugeGraphComponent(CustomComponent):
    """
    A custom component for implementing a graph Store using Huge.
    """

    display_name: str = "HugeGraph (Custom Component)"
    description: str = "Implementation of Graph Store using HugeGraph"
    documentation = "https://python.langchain.com/docs/integrations/graphs/HugeGraph"
    beta = True

    def build_config(self):
        """
        Builds the configuration for the component.

        Returns:
        - dict: A dictionary containing the configuration options for the component.
        """
        return {
            "collection_name": {"display_name": "Collection Name", "value": "langflow"},
            "persist": {"display_name": "Persist"},
            "persist_directory": {"display_name": "Persist Directory"},
            "code": {"show": False, "display_name": "Code"},
            "documents": {"display_name": "Documents", "is_list": True},
            "embedding": {"display_name": "Embedding"},
            "huge_server_cors_allow_origins": {
                "display_name": "Server CORS Allow Origins",
                "advanced": True,
            },
            "huge_server_host": {"display_name": "Server Host", "advanced": True},
            "huge_server_port": {"display_name": "Server Port", "advanced": True},
            "huge_server_grpc_port": {
                "display_name": "Server gRPC Port",
                "advanced": True,
            },
            "huge_server_ssl_enabled": {
                "display_name": "Server SSL Enabled",
                "advanced": True,
            },
        }

    def build(
        self,
        collection_name: str,
        persist: bool,
        huge_server_ssl_enabled: bool,
        persist_directory: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
        documents: Optional[Document] = None,
        huge_server_cors_allow_origins: Optional[str] = None,
        huge_server_host: Optional[str] = None,
        huge_server_port: Optional[int] = None,
        huge_server_grpc_port: Optional[int] = None,
    ) -> Union[GraphStore, BaseRetriever]:
        """
        Builds the Graph Store or BaseRetriever object.

        Args:
        - collection_name (str): The name of the collection.
        - persist_directory (Optional[str]): The directory to persist the Graph Store to.
        - huge_server_ssl_enabled (bool): Whether to enable SSL for the Huge server.
        - persist (bool): Whether to persist the Graph Store or not.
        - embedding (Optional[Embeddings]): The embeddings to use for the Graph Store.
        - documents (Optional[Document]): The documents to use for the Graph Store.
        - huge_server_cors_allow_origins (Optional[str]): The CORS allow origins for the Huge server.
        - huge_server_host (Optional[str]): The host for the Huge server.
        - huge_server_port (Optional[int]): The port for the Huge server.
        - huge_server_grpc_port (Optional[int]): The gRPC port for the Huge server.

        Returns:
        - Union[GraphStore, BaseRetriever]: The Graph Store or BaseRetriever object.
        """

        # Huge settings
        huge_settings = None

        if huge_server_host is not None:
            huge_settings = hugedb.config.Settings(
                huge_server_cors_allow_origins=huge_server_cors_allow_origins
                or None,
                huge_server_host=huge_server_host,
                huge_server_port=huge_server_port or None,
                huge_server_grpc_port=huge_server_grpc_port or None,
                huge_server_ssl_enabled=huge_server_ssl_enabled,
            )

        # If documents, then we need to create a Huge instance using .from_documents
        if documents is not None and embedding is not None:
            return Huge.from_documents(
                documents=documents,  # type: ignore
                persist_directory=persist_directory if persist else None,
                collection_name=collection_name,
                embedding=embedding,
                client_settings=huge_settings,
            )

        return HugeGraph(
            persist_directory=persist_directory, client_settings=huge_settings
        )
