from . import agent_traj  # noqa: F401  — 触发 @register_task 副作用 (phase 5)
from . import bfcl_slice  # noqa: F401  — 触发 @register_task 副作用 (agent_sft phase 1)
from . import iaa_nominal  # noqa: F401  — 触发 @register_task 副作用 (phase 8)
from . import iaa_ordinal  # noqa: F401  — 触发 @register_task 副作用 (phase 8)
from . import mmlu_slice  # noqa: F401  — 触发 @register_task 副作用 (agent_sft phase 1)
from . import mt  # noqa: F401  — 触发 @register_task 副作用
from . import nudge_fire_rate  # noqa: F401  — 触发 @register_task 副作用 (agent_sft phase 1)
from . import qa_open  # noqa: F401  — 触发 @register_task 副作用
from . import rag_qa  # noqa: F401  — 触发 @register_task 副作用 (phase 4)
from . import rag_retrieval  # noqa: F401  — 触发 @register_task 副作用 (phase 4)
from . import safety  # noqa: F401  — 触发 @register_task 副作用 (phase 7)
from . import sentiment_clf  # noqa: F401  — 触发 @register_task 副作用
