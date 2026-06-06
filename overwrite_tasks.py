import importlib
import asyncio
from json import dumps, loads
from dotenv import load_dotenv
from openai import OpenAI
from tlm import TLM
from tlm.config.schema import Config
from tlm.config.presets import ReasoningEffort
from src.tau2.utils.guidance import get_guidance_message, load_guidance
from src.tau2.utils.trustworthiness import trustworthiness_from_messages
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import APICompatibleMessage, SystemMessage
from tau2.environment.environment import Environment
from concurrent.futures import ThreadPoolExecutor
from threading import local

from tau2.utils.llm_utils import to_litellm_messages, to_tau2_messages

load_dotenv()

client = OpenAI()

# Thread-local storage for event loops
_thread_local = local()


def get_event_loop():
    """Get or create an event loop for the current thread."""
    if not hasattr(_thread_local, "loop"):
        _thread_local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_thread_local.loop)
    return _thread_local.loop


DOMAIN = "airline"
SIMULATIONS = f"{DOMAIN}_k4/guidance-1.json"
ENVIRONMENT_MODULE = importlib.import_module(f"src.tau2.domains.{DOMAIN}.environment")
ENVIRONMENT: Environment = ENVIRONMENT_MODULE.get_environment()
AGENT = LLMAgent(
    tools=ENVIRONMENT.get_tools(),
    domain_policy=ENVIRONMENT.get_policy(),
    llm="gpt-4",
)
SYSTEM_MESSAGE = AGENT.system_prompt
SYSTEM_MESSAGES = [SystemMessage(content=SYSTEM_MESSAGE, role="system")]

with open(f"data/tau2/domains/{DOMAIN}/tasks.json", "r") as f:
    tasks = loads(f.read())

with open(f"data/simulations/{SIMULATIONS}", "r") as f:
    simulations = [
        simulation
        for simulation in loads(f.read())["simulations"]
        if simulation["trial"] == 0
    ]
simulations.sort(key=lambda x: x["task_id"])


def worker(allMessages: list[APICompatibleMessage], task_id: str):
    """Worker function that runs in its own thread with its own event loop."""
    # Ensure this thread has an event loop
    loop = get_event_loop()

    i = 0
    tlm = TLM(
        config=Config(
            reasoning_effort=ReasoningEffort.MEDIUM,
        )
    )

    def determine_modification(messages: list[APICompatibleMessage]) -> bool:
        last_message = messages[-1]

        if last_message.role == "assistant" and len(messages) > 1:
            guidance, guidance_message = get_guidance_message(messages[:-1])
            if guidance:
                return True

            # trustworthiness = trustworthiness_from_messages(
            #     messages[:-1], last_message, tools=ENVIRONMENT.get_tools(), tlm=tlm
            # )
            # if trustworthiness["confidence_score"] < 0.75:
            #     return True

        return False

    while i < len(allMessages):
        if determine_modification(allMessages[: i + 1]):
            print(f"Modifying task {task_id} at message index {i}")
            return (task_id, allMessages[:i], True)

        i += 1

    return (task_id, allMessages, False)


if __name__ == "__main__":
    load_guidance(DOMAIN)

    results_dict = {}
    modified_tasks = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for simulation in simulations:
            task_id = simulation["task_id"]
            # if task_id != "35":
            #     continue
            future = executor.submit(
                worker, to_tau2_messages(simulation["messages"]), task_id  # type: ignore
            )
            futures[future] = task_id

        # Collect results with progress reporting
        total_tasks = len(futures)
        for idx, future in enumerate(futures, 1):
            task_id, messages, was_modified = future.result()
            results_dict[task_id] = messages
            if was_modified:
                modified_tasks.append(task_id)
            percent = (idx / total_tasks) * 100
            print(f"Progress: {idx}/{total_tasks} ({percent:.1f}%)")

    for task_id in results_dict.keys():
        task = tasks[int(task_id)]
        if task["initial_state"] == None:
            task["initial_state"] = {}
        # Serialize messages to JSON-compatible format
        serialized_messages = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg
            for msg in results_dict[task_id]
        ]
        task["initial_state"]["message_history"] = serialized_messages
        if task_id in modified_tasks:
            task["modified"] = True
        else:
            task["modified"] = False

    with open(f"data/tau2/domains/{DOMAIN}/tasks.json", "w") as f:
        f.write(dumps(tasks, indent=4))

    print("\nModified the following tasks:")
    for task_id in sorted(modified_tasks, key=int):
        print(f"  - {task_id}")
