# AIFuncKit

## What is it and what can it do?

AIFuncKit- Python framework for AI task orchestration. Probably, sometimes you need, that AI can call some functions, but all what can do this is stuck. It or is slow, or work bad, or even doesn't still work. And AIFuncKit can help you. Out of the box it includes: 
- Decompose a prompt into tasks;
- Select function from user prompt;
- Extract arguments from this;
- Execute method;
- Write down to RAG memory;
- Validate data;
- Protect against prompt injection.

Also, it supports:

- Different variants of execution;
- Levels of logging;
- State management and emergency_halt;
- pretty simple SDK for developer of it;
- Pydantic, Ollama, NumPy, AIOSQLite, ONNX Runtime, different APIs for AI;
- Custom exceptions.

## Code Examples


### 1. Quickstart
Start work with it pretty simple. Define your functions, wrap them in `@engine.action`, and let the orchestrator figure out what needs to be called. Example is below

```python
import asyncio
from aifunckit import AIEngine

engine = AIEngine(
    ai_model="gemma4:e4b", # select AI model
    provider="ollama" # select, where you can download model. Right now, in AIFuncKit is available ollama, anthropic, openai, google gemini (gemini), llama_cpp, hugging face (hf, hugging_face) 
)

# 1. Register simple actions with type hints
@engine.action(description="Get current system resource statistics")
def get_system_stats() -> dict:
    return {"cpu_percent": 14.2, "ram_free_gb": 6.8}

@engine.action(description="Send an alert message to administrator")
def send_alert(message: str) -> str:
    print(f"Notification: {message}")
    return "sent"

async def main():
    await engine.initialize()

    # The model inspects actions, matches parameters, and executes:
    results = await engine.run_engine("Check system health and notify admin if something looks off")
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 2. Building a Plugin (when one function is not enough)
Probably, if you used it, you would have a situation, when you have a lot of functions, and select every of them is not convenient. And here is the decision: you can use plugins. Plugins is a class, which is inherited from BasePlugin class. Here you must set a field `plugin_name`, code your methods and register it into engine. Also, for convenient, you can use `type` field.

```python
import asyncio
from aifunckit import AIEngine, BasePlugin, describe

class BackupPlugin(BasePlugin):
    plugin_name = "backup"

    def __init__(self):
        super().__init__()
        self.last_backup = None

    # method for plugin
    @describe("Create an archive of a specified directory")
    def create_backup(self, directory: str, archive_name: str) -> str:
        self.last_backup = f"/backups/{archive_name}.tar.gz"
        return f"Created {self.last_backup} from {directory}"

    def get_live_context(self) -> dict:
        # Context automatically synced into pipeline environment state
        return {"last_backup": self.last_backup}


async def main():
    engine = AIEngine(provider="ollama")
    engine.register_plugin(BackupPlugin())
    await engine.initialize()

    await engine.run_engine("Make a backup of /var/log into weekly_logs")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 3. Reliability: Circuit Breaker & Emergency Stop
Sometimes, you want to stop pipeline for some reasons. Here I made a emergency halt. You can stop it by `engine.stop_engine()` and before AI destroys everything what it can. To repear it use `engine.resume_engine()` Also, you can give for it how many times exceptions can be raised using `failure_threshgolder`.

```python
import asyncio
from aifunckit import AIEngine
from exceptions.pipeline_exceptions import CircuitBreakerOpenException

engine = AIEngine(
    provider="ollama",
    failure_threshold=3  # Trip the fuse after 3 consecutive errors
)

# You can also manually pull the emergency brake at any time
def handle_danger_signal():
    engine.stop_engine("Hardware overheating signal received!")

async def main():
    await engine.initialize()

    try:
        # If consecutive actions keep failing, the circuit breaker halts the whole pipeline
        # without crashing the main application.
        await engine.run_engine("Attempt to sync remote drive")
    except CircuitBreakerOpenException as exc:
        print(f"Safety engaged: {exc}")

    # Resume when safe
    engine.resume_engine()
```

---

### 4. Scheduled Tasks (Set and forget)
Suppose the system isn't working right now and you want to schedule your task. You can use `engine.schedule_task()` and it will run your prompt later. 

```python
# Runs in an asyncio background worker
task_id = engine.schedule_task(
    prompt="Check disk usage and log if above 90%",
    interval_seconds=300
)

# Later...
engine.unschedule_task(task_id)
```

### 5. Logging
Every software should have a logging system for catching errors. Here you have couple levels of logging.
-  **"silent", "none", "off", "disabled", "0"**. it's not logging anything. It's not recommended by me in production, but if you need it, you can use
- **"critical", "fatal", "error", "1"**. That's logging only critical errors.
- **"warning", "warn", "2"**. That's logging errors and warnings. I recommend it in production.
- **"all", "debug", "verbose", "info", "3"**. That's logging everything. I recommend it when you test your software.

### 6. Other Arguments And Methods Of AIEngine
Here is other argument for configure engine and methods for working with it

#### AIEngine Arguments (`__init__`)

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ai_model` | `str` | `"gemma4:e4b"` | Name of the LLM used for decomposition and argument extraction. |
| `context_window` | `int` | `4096` | Context window size in tokens. |
| `db_path` | `Path \| str` | `Path("aifunckit_runtine")` | Path to the SQLite database for RAG memory and cache. |
| `provider` | `str` | `"ollama"` | AI provider (`"ollama"`, `"google_gemini"`, `"anthropic"`, etc.). |
| `emb_model` | `str` | `"paraphrase-multilingual-MiniLM-L12-v2"` | Embedding model name used by FastEmbed/ONNX. |
| `routing_model` | `str` | `"vector"` | Method routing strategy (`"vector"`, `"text"`, or `"blm"`). |
| `verbose` | `Optional[bool]` | `None` | Enable verbose console logging. |
| `log_to_file` | `Optional[str]` | `"aifunckit.log"` | Log file path (`None` to disable file logging). |
| `log_level` | `Optional[str \| int]` | `None` | Logging level (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`). |
| `failure_threshold` | `int` | `3` | Maximum consecutive errors before Circuit Breaker halts the pipeline. |
| `prompt_injection_protection` | `Optional[List[str]]` | `None` | Additional regex patterns for prompt injection guardrails. |
| `max_prompt_length` | `int` | `4096` | Maximum allowed length of a user prompt. |
| `guard_model_path`  | `str` | `None` | Path to model for sanitizer. |
| `guard_tokenizer_path` | `str` | `None` | Path to tokenizer of sanitizer model. |
| `guard_threshold` | `float` | `0.85` | Threshold for sanitizer model. |

#### AIEngine Methods

| Method | Parameters | Returns | Description |
| :--- | :--- | :--- | :--- |
| `initialize()` | *None* | `None` | Async. Preloads embedding models and initializes SQLite RAG storage. |
| `action(description, name=None)` | `description: str`, `name: Optional[str] = None` | `Decorator` | Decorator to register a standalone function as an executable action. |
| `register_plugin(obj)` | `obj: BasePlugin` | `None` | Registers an instance of a plugin and all its decorated action methods. |
| `delete_plugin(name)` | `name: str` | `None` | Removes a plugin and all its associated actions from the engine. |
| `run_engine(user_prompt)` | `user_prompt: str` | `List[Dict[str, Any]]` | Async. Decomposes prompt, selects topology, and executes matching actions. |
| `get_current_global_context()` | *None* | `dict[str, Any]` | Collects and returns live context dictionaries from all registered plugins. |
| `schedule_task(prompt, interval_seconds, task_id=None)` | `prompt: str`, `interval_seconds: float`, `task_id: Optional[str] = None` | `str` | Registers and starts a recurring background task. Returns `task_id`. |
| `unschedule_task(task_id)` | `task_id: str` | `None` | Stops and removes a recurring scheduled task by its `task_id`. |
| `list_scheduled_tasks()` | *None* | `List[Dict[str, Any]]` | Returns status snapshots of all active scheduled tasks. |
| `stop_engine(reason="Stopped by user")` | `reason: str = "Stopped by user"` | `None` | Emergency brake: trips Circuit Breaker, stops scheduler, blocks runs. |
| `resume_engine()` | *None* | `None` | Resets the Circuit Breaker and restores the engine to operational state. |
| `delete_engine()` | *None* | `None` | Async. Graceful shutdown: closes DB, stops tasks, and frees memory. |

## 7 Exceptions Reference

All custom exceptions inherit from the base framework exception class `__AIBaseException` (or specific domain base classes) to allow granular error handling.

| Category | Exception | Module | When Raised / Description |
| :--- | :--- | :--- | :--- |
| **Plugins & Routing** | `PluginNotFoundError` | `src.exceptions.exceptions` | Raised when vector similarity score is below the confidence threshold or no registered plugin matches the task. |
| | `PluginForDeleteNotFoundError` | `src.exceptions.exceptions` | Raised when attempting to delete or unregister a plugin that does not exist in the registry. |
| | `RoutingModeDoesNotExist` | `src.exceptions.exceptions` | Raised when an unknown routing mode string is provided (supported modes: `vector`, `text`, `blm`). |
| **Security & Guardrails** | `PromptInjectionError` | `src.exceptions.prompt_exeption` | Raised when suspicious prompt injection patterns or adversarial jailbreaks are detected (via regex or ONNX neural guard). |
| | `TooLongPromptError` | `src.exceptions.prompt_exeption` | Raised when the user prompt exceeds the configured `max_prompt_length` limit. |
| **Pipeline & Runtime** | `InvalidPipelineStateException` | `src.exceptions.pipeline_exceptions` | Raised during pre-execution state validation if required context keys, plugins, or metadata are missing. |
| | `CircuitBreakerOpenException` | `src.exceptions.pipeline_exceptions` | Raised by `BreakCircusService` when consecutive execution failures exceed the safety threshold. |
| | `HumansStopPipelineException` | `src.exceptions.pipeline_exceptions` | Raised when an operator or human-in-the-loop signal triggers an emergency stop of the running pipeline. |
| **Vector & Embeddings** | `VectorGenerationError` | `src.exceptions.vector_exceptions` | Raised when text encoding or vector extraction fails in `EmbeddingService`. |
| | `FetchVectorError` | `src.exceptions.vector_exceptions` | Raised when vector retrieval or cosine distance calculation between vectors fails. |
| **Database & Memory** | `DBConnectionError` | `src.exceptions.db_exceptions` | Raised when connecting to or initializing the SQLite runtime memory database fails. |
| | `DBArchiveError` | `src.exceptions.db_exceptions` | Raised when archiving, snapshotting, or persisting memory records encounters a storage fault. |

## Architecture and algorithm of framework

### Algorithm
How it works: a short pipeline
User Prompt ──► Guardrails (Sanitizer) ──► RAG Memory Lookup ──► Decomposition ──► Topology Resolver ──► Pipeline Execution (Sequential / Parallel / Loop / ForEach / Conditional / Timeout).

#### Description
- User Prompt: the prompt you enter, which will serve as the material for the work.
- [Guardrails (Sanitizer)](src/orchestrator/services/sanitizer.py): here system is protected prompt from forbidden patterns or prompt injection and some jailbreaks. It have 2 steps:
    - regex patterns. It's base protection. If you need, you can check it [here](src/orchestrator/services/sanitizer.py#L16-L24)
    - LLM sanitizer. Here I use an LLM to protect from other. If you need it, you should include it in parameters in the AIEngine `guard_threshold`, `guard_tokenizer_path`, `guard_model_path`. More about arguments is [here](README.md#L172-L175).
- [RAG Memory Lookup](src/memory/rag.py): it's memory of model. Here is downloaded all data from decomposition and will use model for work. It can help model work with your prompt and task. Data from here is using on [prompt decomposition](README.md#L206) and [extracting arguments](README.md#L210).
- [Decomposition](src/orchestrator/services/): it's the most important part of pipeline. Here prompt is being processed this stage divide into several substages:
    - [prompt decomposition](src/orchestrator/services/prompt_decomposition.py). Here prompt is divided by tasks.
    - [plugin routing](src/orchestrator/services/plugin_routing.py). Here system is selected method or function for executing. It select callable object and push to the next substage. It can do it by 2 methods:
        - Vector routing (by default). It use NumPy matrix for select and encoded by fastembed.
        - BLM routing. It use BLM model for select. 
    - [extracting arguments](src/orchestrator/services/argument_extraction.py). This substage is extracting arguments using LLM. It also use memory for more correct work. Also, It need types for correct work. And for this reason you must use type hints.
- [Topology resolver](src/orchestrator/topology_resolver.py). It's select, how it will execute. This can be done in the following way:
    - [Sequential](src/orchestrator/pipelines/sequential.py)
    - [Parallel](src/orchestrator/pipelines/parallel.py)
    - [Conditional](src/orchestrator/pipelines/conditional.py)
    - [Loop](src/orchestrator/pipelines/loop.py)
    - [For each loop](src/orchestrator/pipelines/for_each_loop.py)
    - [Timeout](src/orchestrator/pipelines/timeout.py)
    - [Schedule](src/orchestrator/pipelines/schedule.py)
- [Pipeline Execution](src/orchestrator/pipelines/import_strategies/): Here pipeline is executed methods. It have the same variants of executing as topology resolver.

### Architecture of project
```
AIFuncKit/
├── src/
│   ├── engine.py                     # Main entry point (AIEngine)
│   ├── sdk.py                        # Plugin SDK (BasePlugin, @decribe, @description)
│   ├── log_config.py
│   ├── orchestrator/                 # Execution core & pipeline routing
│   │   ├── pipeline.py               # MasterOrchestrationEngine
│   │   ├── topology_resolver.py      # Pipeline topology matcher (linear, parallel, loops)
│   │   ├── general_methods.py
│   │   ├── pipelines/                # Execution strategies (linear, parallel, for_each, schedule)
│   │   └── services/                 # Core subsystems
│   │       ├── sanitizer.py          # Prompt injection guard (Regex + ONNX)
│   │       ├── plugin_routing.py     # BLAS vector & keyword task-to-plugin router
│   │       ├── prompt_decomposition.py
│   │       ├── argument_extraction.py
│   │       ├── action_execution.py
│   │       ├── break_circus.py       # Circuit breaker
│   │       └── state_manager.py      # Context snapshotting & rollback
│   ├── memory/                       # Vector memory & RAG storage (SQLite + FastEmbed ONNX)
│   ├── model_provide/                # LLM providers (OpenAI, Ollama, Hugging Face, etc)
│   ├── schemas/                      # Pydantic data contracts & PipelineContext
│   └── exceptions/
└── pyproject.toml
```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details. 