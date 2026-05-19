import importlib
import asyncio
from json import dumps, loads
from dotenv import load_dotenv
from openai import OpenAI
from tlm import TLM
from tlm.config.base import ConfigInput
from src.tau2.utils.guidance import get_guidance_message
from src.tau2.utils.trustworthiness import trustworthiness_from_messages
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import APICompatibleMessage, SystemMessage
from tau2.environment.environment import Environment
from concurrent.futures import ThreadPoolExecutor
from threading import local

from tau2.utils.llm_utils import to_tau2_messages

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
SIMULATIONS = f"{DOMAIN}_k1/llm.json"
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
    simulations = loads(f.read())["simulations"]
simulations.sort(key=lambda x: x["task_id"])


if __name__ == "__main__":
    for task in tasks:
        del task["initial_state"]["message_history"]

    with open(f"data/tau2/domains/{DOMAIN}/tasks.json", "w") as f:
        f.write(dumps(tasks, indent=4))
