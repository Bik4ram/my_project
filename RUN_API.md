# Running the API locally

```bash
export SEMRAG_DATABASE_URL="postgresql+psycopg2://semrag:semrag@localhost:5432/semrag"
export SEMRAG_API_KEYS="your-dev-key-here"
alembic upgrade head

uvicorn semrag.api.app:app --reload --port 8000
```

Then:
```bash
curl -X POST http://localhost:8000/v1/projects/myproj/index \
  -H "Authorization: Bearer your-dev-key-here" \
  -H "Content-Type: application/json" \
  -d '{"root_path": "/path/to/your/project"}'

curl -X POST http://localhost:8000/v1/projects/myproj/query \
  -H "Authorization: Bearer your-dev-key-here" \
  -H "Content-Type: application/json" \
  -d '{"query": "why does the start condition assertion fail"}'
```

`/health` needs no auth. Full interactive docs at `http://localhost:8000/docs`.

## LLM / Agent layer (Parts 13-14 - written but not live-tested)

`src/semrag/reasoning/llm/openai_compatible_llm.py` targets the standard
`/v1/chat/completions` contract (Ollama, vLLM, llama.cpp) with real
streaming + tool-call support, but was never run against a live model
server in this sandbox (network-blocked, same as Part 9's embedding
provider). Logic is tested via mocked HTTP
(`tests/unit/llm/test_openai_compatible_llm.py`).

`src/semrag/reasoning/agents/` implements a tool-calling agent loop
(`AgentLoop`) with three real tools (`search_codebase`, `read_file`,
`list_files`). Tested via a scripted fake LLM
(`tests/unit/agents/test_agent_loop.py`) plus a full integration test
using the real retrieval pipeline
(`tests/unit/agents/test_agent_retrieval_integration.py`).

To wire it to a real model:
```python
from semrag.reasoning.llm.openai_compatible_llm import OpenAICompatibleLLM
from semrag.reasoning.agents.agent_loop import AgentLoop
from semrag.reasoning.agents.tools import ToolRegistry, SearchCodebaseTool, ReadFileTool, ListFilesTool

llm = OpenAICompatibleLLM(base_url="http://localhost:11434", model="qwen2.5")
tools = ToolRegistry([SearchCodebaseTool(...), ReadFileTool(...), ListFilesTool(...)])
loop = AgentLoop(llm, tools)
result = loop.run([{"role": "user", "content": "why does X fail?"}])
```
