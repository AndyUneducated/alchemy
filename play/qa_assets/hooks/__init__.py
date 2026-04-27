"""qa_assets.hooks: deterministic stage hooks for the QA supervisor workflow.

Industry-standard layout (one function per file when each is non-trivial,
all re-exported here for convenience). Workflow's ``hooks_module:
qa_assets.hooks`` resolves bare ``fn:`` names against this package.
"""

from .load_csv import load_csv
from .load_each_prd import load_each_prd
from .to_yaml import to_yaml

__all__ = ["load_csv", "load_each_prd", "to_yaml"]
