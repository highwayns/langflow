from typing import Optional, Union
from langflow import CustomComponent

from langchain.graphs import NeptuneGraph
from langchain.schema import Document
from langchain.graphstores.base import GraphStore
from langchain.schema import BaseRetriever
from langchain.embeddings.base import Embeddings


class NeptuneGraphComponent(CustomComponent):
    """
    A custom component for implementing a graph Store using Neptune.
    """

    display_name: str = "NeptuneGraph (Custom Component)"
    description: str = "Implementation of Graph Store using NeptuneGraph"
    documentation = "https://python.langchain.com/docs/integrations/graphs/NeptuneGraph"
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
            "neptune_server_cors_allow_origins": {
                "display_name": "Server CORS Allow Origins",
                "advanced": True,
            },
            "neptune_server_host": {"display_name": "Server Host", "advanced": True},
            "neptune_server_port": {"display_name": "Server Port", "advanced": True},
            "neptune_server_grpc_port": {
                "display_name": "Server gRPC Port",
                "advanced": True,
            },
            "neptune_server_ssl_enabled": {
                "display_name": "Server SSL Enabled",
                "advanced": True,
            },
        }

    def build(
        self,
        collection_name: str,
        persist: bool,
        neptune_server_ssl_enabled: bool,
        persist_directory: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
        documents: Optional[Document] = None,
        neptune_server_cors_allow_origins: Optional[str] = None,
        neptune_server_host: Optional[str] = None,
        neptune_server_port: Optional[int] = None,
        neptune_server_grpc_port: Optional[int] = None,
    ) -> Union[GraphStore, BaseRetriever]:
        """
        Builds the Graph Store or BaseRetriever object.

        Args:
        - collection_name (str): The name of the collection.
        - persist_directory (Optional[str]): The directory to persist the Graph Store to.
        - neptune_server_ssl_enabled (bool): Whether to enable SSL for the Neptune server.
        - persist (bool): Whether to persist the Graph Store or not.
        - embedding (Optional[Embeddings]): The embeddings to use for the Graph Store.
        - documents (Optional[Document]): The documents to use for the Graph Store.
        - neptune_server_cors_allow_origins (Optional[str]): The CORS allow origins for the Neptune server.
        - neptune_server_host (Optional[str]): The host for the Neptune server.
        - neptune_server_port (Optional[int]): The port for the Neptune server.
        - neptune_server_grpc_port (Optional[int]): The gRPC port for the Neptune server.

        Returns:
        - Union[GraphStore, BaseRetriever]: The Graph Store or BaseRetriever object.
        """

        # Neptune settings
        neptune_settings = None

        if neptune_server_host is not None:
            neptune_settings = neptunedb.config.Settings(
                neptune_server_cors_allow_origins=neptune_server_cors_allow_origins
                or None,
                neptune_server_host=neptune_server_host,
                neptune_server_port=neptune_server_port or None,
                neptune_server_grpc_port=neptune_server_grpc_port or None,
                neptune_server_ssl_enabled=neptune_server_ssl_enabled,
            )

        # If documents, then we need to create a Neptune instance using .from_documents
        if documents is not None and embedding is not None:
            return Neptune.from_documents(
                documents=documents,  # type: ignore
                persist_directory=persist_directory if persist else None,
                collection_name=collection_name,
                embedding=embedding,
                client_settings=neptune_settings,
            )

        return NeptuneGraph(
            persist_directory=persist_directory, client_settings=neptune_settings
        )
