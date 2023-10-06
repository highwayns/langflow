from typing import Any, Dict, List, Optional, Type

from langchain import graphs

from langflow.interface.base import LangChainTypeCreator
from langflow.interface.importing.utils import import_class
from langflow.services.getters import get_settings_manager

from langflow.template.frontend_node.graphstores import GraphStoreFrontendNode
from loguru import logger
from langflow.utils.util import build_template_from_method


class GraphstoreCreator(LangChainTypeCreator):
    type_name: str = "graphstores"

    @property
    def frontend_node_class(self) -> Type[GraphStoreFrontendNode]:
        return GraphStoreFrontendNode

    @property
    def type_to_loader_dict(self) -> Dict:
        if self.type_dict is None:
            self.type_dict: dict[str, Any] = {
                graphstore_name: import_class(
                    f"langchain.graphs.{graphstore_name}"
                )
                for graphstore_name in graphs.__all__
            }
        return self.type_dict

    def get_signature(self, name: str) -> Optional[Dict]:
        """Get the signature of an embedding."""
        try:
            return build_template_from_method(
                name,
                type_to_cls_dict=self.type_to_loader_dict,
                method_name="from_texts",
            )
        except ValueError as exc:
            raise ValueError(f"Graph Store {name} not found") from exc
        except AttributeError as exc:
            logger.error(f"Graph Store {name} not loaded: {exc}")
            return None

    def to_list(self) -> List[str]:
        settings_manager = get_settings_manager()
        return [
            graphstore
            for graphstore in self.type_to_loader_dict.keys()
            if graphstore in settings_manager.settings.GRAPHSTORES
            or settings_manager.settings.DEV
        ]


graphstore_creator = GraphstoreCreator()
