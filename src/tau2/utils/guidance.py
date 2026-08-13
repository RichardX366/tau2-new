from asyncio import gather, get_event_loop

from litellm import acompletion
from openai.types.chat import ChatCompletionMessageParam
from json import dumps, load

from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
)
from tau2.utils.llm_utils import to_litellm_messages

all_guidance: list[dict] = None  # type: ignore


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

    request = get_event_loop().run_until_complete(
        gather(
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
    )

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
