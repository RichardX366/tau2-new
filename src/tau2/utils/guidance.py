import asyncio
import threading
from concurrent.futures import Future
from json import dumps, load
from typing import Any, Coroutine, TypeVar

import litellm
from litellm import acompletion
from litellm.caching.caching import Cache
from openai.types.chat import ChatCompletionMessageParam

from tau2.config import (
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
)
from tau2.utils.llm_utils import to_litellm_messages

all_guidance: list[dict] = None  # type: ignore

T = TypeVar("T")


class _GuidanceEventLoop:
    """Own the single event loop used by asynchronous guidance requests."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._lock = threading.Lock()
        self._cache: Cache | None = None
        self._previous_cache: Any = None
        self._cache_active = False

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._started.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            asyncio.set_event_loop(None)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._started.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="tau2-guidance-event-loop",
                daemon=True,
            )
            self._thread.start()
        self._started.wait()

    def run(self, coroutine: Coroutine[object, object, T]) -> T:
        self.start()
        loop = self._loop
        if loop is None:
            coroutine.close()
            raise RuntimeError("Guidance event loop failed to start")
        future: Future[T] = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return future.result()

    async def _drain_request_tasks(self) -> None:
        """Await finite LiteLLM work while leaving its logging worker alive."""
        current = asyncio.current_task()
        pending = []
        for task in asyncio.all_tasks():
            coroutine_name = task.get_coro().__qualname__
            if (
                task is not current
                and not task.done()
                and coroutine_name != "LoggingWorker._worker_loop"
            ):
                pending.append(task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _activate_cache(self) -> None:
        if self._cache_active:
            return
        if self._cache is None:
            self._cache = Cache(
                type="redis",
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                namespace=(
                    f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm:guidance"
                ),
                ttl=REDIS_CACHE_TTL,
            )
        self._previous_cache = litellm.cache
        litellm.cache = self._cache
        self._cache_active = True

    def activate_cache(self) -> None:
        """Install the guidance-only cache before starting guidance requests."""
        self.run(self._activate_cache())

    async def _deactivate_cache(self) -> None:
        if not self._cache_active:
            return
        await self._drain_request_tasks()
        if litellm.cache is self._cache:
            litellm.cache = self._previous_cache
        self._previous_cache = None
        self._cache_active = False

    def deactivate_cache(self) -> None:
        """Drain guidance cache writes and restore Tau2's global cache."""
        self.run(self._deactivate_cache())

    def shutdown(self) -> None:
        """Cancel and await background work before closing the owned loop."""
        with self._lock:
            loop = self._loop
            thread = self._thread
            if loop is None or thread is None:
                return

        async def close_cache_and_cancel_pending_tasks() -> None:
            await self._deactivate_cache()
            if self._cache is not None:
                await self._cache.disconnect()
                self._cache = None
            current = asyncio.current_task()
            pending = [
                task
                for task in asyncio.all_tasks()
                if task is not current and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(
            close_cache_and_cancel_pending_tasks(), loop
        ).result()
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        with self._lock:
            self._loop = None
            self._thread = None


_guidance_event_loop = _GuidanceEventLoop()


def activate_guidance_cache() -> None:
    """Use a Redis cache owned exclusively by the guidance event loop."""
    _guidance_event_loop.activate_cache()


def deactivate_guidance_cache() -> None:
    """Finish guidance cache work and restore Tau2's cache."""
    _guidance_event_loop.deactivate_cache()


def shutdown_guidance_event_loop() -> None:
    """Shut down LiteLLM work scheduled by the guidance subsystem."""
    _guidance_event_loop.shutdown()


GUIDANCE_DETERMINATION_PROMPT = """
You are an expert assistant that determines whether a certain question is true about the query you are provided.
If you are provided with multiple messages, the query only applies to the final message labeled "<QUERY_MESSAGE/>", with the other messages just serving as context.
You are only to respond with the word "Yes" or "No" and ABSOLUTELY NOTHING ELSE."""


def format_messages_to_string(
    messages: list[ChatCompletionMessageParam], lastMessageIndicator=""
) -> str:
    """Format messages for inclusion in a prompt. Assumes messages has at least one message."""
    messages_str = ""
    for i, message in enumerate(messages):
        messages_str += "<MESSAGE>\n"
        if lastMessageIndicator and i == len(messages) - 1:
            messages_str += lastMessageIndicator + "\n"
        messages_str += message["role"].capitalize() + ":\n"
        messages_str += message["content"] or ""  # type: ignore
        if message.get("content", "") and message.get("tool_calls", []):
            messages_str += "\n\n"
        if message.get("tool_calls", []):
            messages_str += dumps(message.get("tool_calls"), indent=2)
        messages_str += "\n</MESSAGE>\n\n"
    return messages_str


async def false():
    return False


def get_last_n_messages(messages: list[ChatCompletionMessageParam], n: int):
    to_return = []
    i = len(messages) - 1
    while n > 0:
        m = messages[i]
        if m["role"] != "tool":
            n -= 1
        to_return = [m] + to_return
        i -= 1
        if i < 0:
            break
    return to_return


def consult_ai_guidance(
    messages: list[ChatCompletionMessageParam],
    triggers="before",
    previous_guidance: set[str] = set(),
) -> list[str]:

    async def determine_guidance_relevance(guidance: dict) -> bool:
        if guidance["triggers"] == "before":
            if messages[-1]["role"] == "tool":
                if guidance["type"] == "text":
                    return False
            else:
                if guidance["type"] == "tool":
                    return False
        else:
            if guidance["type"] == "tool":
                if not messages[-1].get("tool_calls", []):
                    return False
            elif guidance["type"] == "text":
                if not messages[-1]["content"]:  # type: ignore
                    return False

        query = format_messages_to_string(
            get_last_n_messages(messages, guidance["numMessages"]), "<QUERY_MESSAGE/>"
        )

        determination_response = await acompletion(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": GUIDANCE_DETERMINATION_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"""<QUERY>
{query}
</QUERY>

<QUESTION>
{guidance["query"]}
</QUESTION>""",
                },
            ],
        )

        return (
            "yes" in determination_response.choices[0].message.content.strip().lower()  # type: ignore
        )

    async def determine_all_guidance() -> list[bool]:
        return await asyncio.gather(
            *[
                (
                    determine_guidance_relevance(guidance)
                    if guidance["triggers"] == triggers
                    and guidance["guidance"] not in previous_guidance
                    else false()
                )
                for guidance in all_guidance
            ]
        )

    request = _guidance_event_loop.run(determine_all_guidance())

    guidance_returned = [
        guidance["guidance"]
        for guidance, determination in zip(all_guidance, request)
        if determination
    ]

    # print(f"Returned {len(guidance_returned)} guidance entries")

    return guidance_returned


def load_guidance(domain: str):
    global all_guidance

    if all_guidance is not None:
        return

    GUIDANCE_FILE = f"data/tau2/domains/{domain}/guidance.json"

    try:
        with open(GUIDANCE_FILE, "r") as f:
            all_guidance = load(f)
    except FileNotFoundError:
        all_guidance = []
    for guidance in all_guidance:
        if "triggers" not in guidance:
            guidance["triggers"] = "before"
        if "type" not in guidance:
            guidance["type"] = "text"
        if "numMessages" not in guidance:
            guidance["numMessages"] = 1
        if "permanent" not in guidance:
            guidance["permanent"] = False


def get_guidance_dict(guidance: str):
    for g in all_guidance:
        if g["guidance"] == guidance:
            return g
    return {}


def get_cancel_tool_messages(assistant_message: AssistantMessage) -> list[ToolMessage]:
    canceled_tool_messages = []
    if assistant_message.tool_calls:
        canceled_tool_messages = [
            ToolMessage(
                id=tool_call.id,
                role="tool",
                content="Tool Call Canceled",
                requestor="assistant",
            )
            for tool_call in assistant_message.tool_calls
        ]
    return canceled_tool_messages


def get_pre_guidance_message(messages: list[APICompatibleMessage]):
    """
    Given the message history, determine which guidance statements are relevant and should be included in the system prompt, and return those along with the original messages.

    Returns:
    - guidance: list of relevant guidance statements (strings)
    - guidance_messages: list of SystemMessage objects containing the relevant guidance to be included in the system
    """

    previous_guidance: set[str] = {
        g
        for m in messages
        if isinstance(m, AssistantMessage) and m.raw_data
        for g in m.raw_data.get("guidance", [])
        if get_guidance_dict(g)["permanent"]
    }

    guidance = consult_ai_guidance(to_litellm_messages(messages), previous_guidance=previous_guidance)  # type: ignore

    total_guidance = set(guidance) | previous_guidance

    return list(set(guidance)), (
        [
            SystemMessage(
                role="system",
                content="Guidance for the next assistant response. You must follow the rules that apply. If they don't apply, you can ignore them:\n"
                + "\n".join([f"- {g}" for g in total_guidance]),
            )
        ]
        if total_guidance
        else []
    )


def get_post_guidance_message(messages: list[APICompatibleMessage]):
    """
    Given the message history, determine which guidance statements are relevant and should be included in the system prompt, and return those along with the original messages.

    Returns:
    - guidance: list of relevant guidance statements (strings)
    - guidance_messages: list of SystemMessage objects containing the relevant guidance to be included in the system
    """

    previous_guidance: set[str] = {
        g
        for m in messages
        if isinstance(m, AssistantMessage) and m.raw_data
        for g in m.raw_data.get("guidance", [])
        if get_guidance_dict(g)["permanent"]
    }

    guidance = consult_ai_guidance(to_litellm_messages(messages), triggers="after", previous_guidance=previous_guidance)  # type: ignore

    if not guidance:
        return [], []

    return list(set(guidance)), (
        [
            SystemMessage(
                role="system",
                content=(
                    f"""There may have been some issues with your previous message. You must rewrite it to address the following guidance. You must follow the rules that apply. If they don't apply, you can ignore them. If none apply or no changes are needed, just repeat your previous message verbatim:
{"\n".join([f"- {g}" for g in guidance])}"""
                    + (
                        f"""

Be sure that if you rewrite your message, you still adhere to the following guidance as well. If they don't apply, you can ignore them:
{"\n".join([f"- {g}" for g in previous_guidance])}"""
                        if previous_guidance
                        else ""
                    )
                ),
            )
        ]
    )
