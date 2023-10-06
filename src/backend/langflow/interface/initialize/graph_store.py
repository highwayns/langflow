from typing import Any, Callable, Dict, Type
from langchain.graphs import (
    Neo4jGraph,
    NebulaGraph,
    NeptuneGraph,
    KuzuGraph,
    HugeGraph,
    ArangoGraph,
)

import os

import orjson


def docs_in_params(params: dict) -> bool:
    """Check if params has documents OR texts and one of them is not an empty list,
    If any of them is not an empty list, return True, else return False"""
    return ("documents" in params and params["documents"]) or (
        "texts" in params and params["texts"]
    )


def initialize_neo4jgraph(class_object: Type[Neo4jGraph], params: dict):
    """Initialize Neo4jGraph and return the class object"""

    return class_object.from_documents(**params)

def initialize_nebulagraph(class_object: Type[NebulaGraph], params: dict):
    """Initialize NebulaGraph and return the class object"""

    return class_object.from_documents(**params)

def initialize_neptunegraph(class_object: Type[NeptuneGraph], params: dict):
    """Initialize NeptuneGraph and return the class object"""

    return class_object.from_documents(**params)

def initialize_kuzugraph(class_object: Type[KuzuGraph], params: dict):
    """Initialize KuzuGraph and return the class object"""

    return class_object.from_documents(**params)

def initialize_hugegraph(class_object: Type[HugeGraph], params: dict):
    """Initialize HugeGraph and return the class object"""

    return class_object.from_documents(**params)

def initialize_arangograph(class_object: Type[ArangoGraph], params: dict):
    """Initialize ArangoGraph and return the class object"""

    return class_object.from_documents(**params)




grastore_initializer: Dict[str, Callable[[Type[Any], dict], Any]] = {
    "Neo4jGraph": initialize_neo4jgraph,
    "NebulaGraph": initialize_nebulagraph,
    "NeptuneGraph": initialize_neptunegraph,
    "KuzuGraph": initialize_kuzugraph,
    "HugeGraph": initialize_hugegraph,
    "ArangoGraph": initialize_arangograph,
}
