from typing import Optional, Union
from langflow import CustomComponent

from langchain.graphs import NebulaGraph
from langchain.schema import Document
from langchain.graphstores.base import GraphStore
from langchain.schema import BaseRetriever
from langchain.embeddings.base import Embeddings


class NebulaGraphComponent(CustomComponent):
    """
    A custom component for implementing a graph Store using Nebula.
    """

    display_name: str = "NebulaGraph (Custom Component)"
    description: str = "Implementation of Graph Store using NebulaGraph"
    documentation = "https://python.langchain.com/docs/integrations/graphs/NebulaGraph"
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
            "nebula_server_cors_allow_origins": {
                "display_name": "Server CORS Allow Origins",
                "advanced": True,
            },
            "nebula_server_host": {"display_name": "Server Host", "advanced": True},
            "nebula_server_port": {"display_name": "Server Port", "advanced": True},
            "nebula_server_grpc_port": {
                "display_name": "Server gRPC Port",
                "advanced": True,
            },
            "nebula_server_ssl_enabled": {
                "display_name": "Server SSL Enabled",
                "advanced": True,
            },
        }

    def build(
        self,
        collection_name: str,
        persist: bool,
        nebula_server_ssl_enabled: bool,
        persist_directory: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
        documents: Optional[Document] = None,
        nebula_server_cors_allow_origins: Optional[str] = None,
        nebula_server_host: Optional[str] = None,
        nebula_server_port: Optional[int] = None,
        nebula_server_grpc_port: Optional[int] = None,
    ) -> Union[GraphStore, BaseRetriever]:
        """
        Builds the Graph Store or BaseRetriever object.

        Args:
        - collection_name (str): The name of the collection.
        - persist_directory (Optional[str]): The directory to persist the Graph Store to.
        - nebula_server_ssl_enabled (bool): Whether to enable SSL for the Nebula server.
        - persist (bool): Whether to persist the Graph Store or not.
        - embedding (Optional[Embeddings]): The embeddings to use for the Graph Store.
        - documents (Optional[Document]): The documents to use for the Graph Store.
        - nebula_server_cors_allow_origins (Optional[str]): The CORS allow origins for the Nebula server.
        - nebula_server_host (Optional[str]): The host for the Nebula server.
        - nebula_server_port (Optional[int]): The port for the Nebula server.
        - nebula_server_grpc_port (Optional[int]): The gRPC port for the Nebula server.

        Returns:
        - Union[GraphStore, BaseRetriever]: The Graph Store or BaseRetriever object.
        """

        # Nebula settings
        nebula_settings = None

        if nebula_server_host is not None:
            nebula_settings = nebuladb.config.Settings(
                nebula_server_cors_allow_origins=nebula_server_cors_allow_origins
                or None,
                nebula_server_host=nebula_server_host,
                nebula_server_port=nebula_server_port or None,
                nebula_server_grpc_port=nebula_server_grpc_port or None,
                nebula_server_ssl_enabled=nebula_server_ssl_enabled,
            )

        # If documents, then we need to create a Nebula instance using .from_documents
        if documents is not None and embedding is not None:
            return Nebula.from_documents(
                documents=documents,  # type: ignore
                persist_directory=persist_directory if persist else None,
                collection_name=collection_name,
                embedding=embedding,
                client_settings=nebula_settings,
            )

        return NebulaGraph(
            persist_directory=persist_directory, client_settings=nebula_settings
        )
