import importlib
import asyncio
from json import dumps, loads
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tlm import TLM
from tlm.config.schema import Config
from tlm.config.presets import ReasoningEffort
from src.tau2.utils.guidance import (
    get_post_guidance_message,
    get_pre_guidance_message,
    load_guidance,
)
from tau2.agent.llm_agent import LLMAgent
from tau2.config import DEFAULT_MAX_STEPS
from tau2.data_model.message import APICompatibleMessage, SystemMessage
from tau2.data_model.simulation import Results, TextRunConfig
from tau2.environment.environment import Environment
from concurrent.futures import ThreadPoolExecutor
from threading import local
from tau2.runner.batch import run_domain
from tau2.utils.llm_utils import to_tau2_messages
from os import remove, removedirs

load_dotenv()

client = OpenAI()

# Thread-local storage for event loops
_thread_local = local()

# Constants

DOMAIN = "retail"
ALREADY_GUIDANCE = 0

SIMULATIONS = f"{DOMAIN}_k4/guidance-{ALREADY_GUIDANCE}.json"
SAVE_TO = f"data/simulations/{DOMAIN}_k4/guidance-{ALREADY_GUIDANCE + 1}.json"

# Code


def get_event_loop():
    """Get or create an event loop for the current thread."""
    if not hasattr(_thread_local, "loop"):
        _thread_local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_thread_local.loop)
    return _thread_local.loop


load_guidance(DOMAIN)

from src.tau2.utils.guidance import all_guidance

ENVIRONMENT_MODULE = importlib.import_module(f"src.tau2.domains.{DOMAIN}.environment")
ENVIRONMENT: Environment = ENVIRONMENT_MODULE.get_environment()
AGENT = LLMAgent(
    tools=ENVIRONMENT.get_tools(),
    domain_policy=ENVIRONMENT.get_policy(),
    llm="gpt-4",
)
SYSTEM_MESSAGE = AGENT.system_prompt
SYSTEM_MESSAGES = [SystemMessage(content=SYSTEM_MESSAGE, role="system")]
USED_GUIDANCE = all_guidance[:ALREADY_GUIDANCE]

config = TextRunConfig(
    domain=DOMAIN,
    task_set_name=None,
    task_split_name="base",
    task_ids=None,
    num_tasks=None,
    llm_args_user={"temperature": 0.0, "reasoning_effort": "low"},
    llm_args_agent={"temperature": 0.0, "reasoning_effort": "medium"},
    user="user_simulator",
    max_steps=DEFAULT_MAX_STEPS,
    enforce_communication_protocol=False,
    num_trials=1,
    max_errors=10,
    timeout=None,
    save_to="temp",
    max_concurrency=10,
    seed=300,
    log_level="ERROR",
    verbose_logs=False,
    max_retries=3,
    retry_delay=1.0,
    auto_resume=False,
    auto_review=False,
    review_mode="full",
    hallucination_retries=3,
    retrieval_config=None,
    retrieval_config_kwargs=None,
    is_remote=False,
    text_streaming_config=None,
    agent="guidance_agent",
    llm_agent="gpt-5.2",
    llm_user="gpt-5.2",
)

with open(f"data/tau2/domains/{DOMAIN}/tasks.json", "r") as f:
    tasks = loads(f.read())

with open(f"data/simulations/{SIMULATIONS}", "r") as f:
    json = loads(f.read())
    trials = json["info"]["num_trials"]
    all_simulations = [simulation for simulation in json["simulations"]]
all_simulations.sort(key=lambda x: x["task_id"])


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
            guidance, guidance_message = get_pre_guidance_message(messages[:-1])
            post_guidance, post_guidance_message = get_post_guidance_message(messages)
            for used in USED_GUIDANCE:
                if used["guidance"] in guidance:
                    guidance.remove(used["guidance"])
                if used["guidance"] in post_guidance:
                    post_guidance.remove(used["guidance"])
            if guidance or post_guidance:
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


def modify_tasks(trial: int):
    """Main function to modify tasks based on simulations."""
    simulations = [
        simulation for simulation in all_simulations if simulation["trial"] == trial
    ]
    results_dict = {}
    modified_tasks = []

    print(f"Processing trial {trial} with {len(simulations)} simulations...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for simulation in simulations:
            task_id = simulation["task_id"]
            future = executor.submit(
                worker, to_tau2_messages(simulation["messages"]), task_id  # type: ignore
            )
            futures[future] = task_id

        for idx, future in enumerate(futures, 1):
            task_id, messages, was_modified = future.result()
            results_dict[task_id] = messages
            if was_modified:
                modified_tasks.append(task_id)

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

    return modified_tasks


if __name__ == "__main__":
    results: list[Results] = []

    Path("data/simulations/temp/results.json").unlink(missing_ok=True)

    for trial in range(trials):
        modified = modify_tasks(trial)
        result = run_domain(config)
        remove("data/simulations/temp/results.json")
        removedirs("data/simulations/temp")
        for sim in result.simulations:
            if sim.info is None:
                sim.info = {}
            sim.info["modified"] = sim.task_id in modified
            sim.trial = trial
        results += [result]

    result = Results(
        info=results[0].info,
        simulation_index=None,
        tasks=results[0].tasks,
        timestamp=results[0].timestamp,
        simulations=[sim for r in results for sim in r.simulations],
    )
    result.info.num_trials = trials

    result.save(Path(SAVE_TO))

    with open(SAVE_TO, "r") as f:
        json = loads(f.read())
        json["path"] = str("../tau2-new/" + SAVE_TO)
    with open(SAVE_TO, "w") as f:
        f.write(dumps(json, indent=2))

    for task in tasks:
        if "message_history" in task["initial_state"]:
            del task["initial_state"]["message_history"]
        if not task["initial_state"]:
            task["initial_state"] = None
        if "modified" in task:
            del task["modified"]

    with open(f"data/tau2/domains/{DOMAIN}/tasks.json", "w") as f:
        f.write(dumps(tasks, indent=4))
