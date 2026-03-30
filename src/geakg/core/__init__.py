"""Core abstractions for GEAKG generalization.

RoleSchema is the central abstraction that decouples the framework from
any specific set of roles. Each case study (optimization, NAS) provides
its own RoleSchema implementation.

CaseStudy bundles a RoleSchema with domain config, base operators, and
MetaGraph factory into a single object.
"""

from src.geakg.core.role_schema import RoleSchema
from src.geakg.core.case_study import CaseStudy

__all__ = [
    "CaseStudy",
    "RoleSchema",
]
