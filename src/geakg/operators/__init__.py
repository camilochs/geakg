"""NS-SE operator wrappers and adapters.

This module provides wrappers to integrate external optimization code
(like LLaMEA evolved functions) as NS-SE operators that participate in
pheromone-based selection and can be transferred to other domains.
"""

from .llamea_wrapper import LLaMEAOperatorWrapper, load_llamea_as_operator

__all__ = ["LLaMEAOperatorWrapper", "load_llamea_as_operator"]
