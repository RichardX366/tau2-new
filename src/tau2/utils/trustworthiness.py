from copy import deepcopy
from json import dumps, loads
import time
import uuid

from litellm.cost_calculator import cost_per_token
from tlm import TLM
from tlm.inference import InferenceResult

from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
)
from openai.types.chat import ChatCompletion

from tau2.environment.tool import Tool
from tau2.utils.llm_utils import to_litellm_messages


TLM_MODEL = "gpt-4.1-mini"


def message_to_chat_completion(
    msg: AssistantMessage, model="dummy-model-v1"
) -> ChatCompletion:
    """
    Convert your internal message schema into a valid ChatCompletion API object.
    """

    message = loads(dumps(msg.model_dump()))

    if message.get("tool_calls") is not None:
        toolCalls = message.get("tool_calls")
        for tc in toolCalls:
            tc["function"] = {
                "name": tc["name"],
                "arguments": dumps(tc["arguments"]),
            }
            tc["type"] = "function"
            del tc["name"]
            del tc["arguments"]
    else:
        toolCalls = None

    usage = message.get("usage", {}) or {}

    # AUTO-FILL usage if missing
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    # BUILD ChatCompletion FORMAT
    chatCompletion = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": message.get("content"),
                },
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    if toolCalls is not None:
        chatCompletion["choices"][0]["message"]["tool_calls"] = toolCalls

    return chatCompletion  # type: ignore


def trustworthiness_from_messages(
    messages: list[APICompatibleMessage],
    assistant_message: AssistantMessage,
    tools: list[Tool],
    tlm: TLM,
) -> InferenceResult:
    """Calculate the trustworthiness of the response based on the messages."""

    if assistant_message.raw_data is None:
        assistant_message.raw_data = {}

    review_messages = deepcopy(messages)

    tool_strings = "\n".join([dumps(t.openai_schema) for t in tools])

    review_messages.insert(
        1,
        SystemMessage(
            role="system",
            content=f"""The assistant has access to the following tools:
<tools>
{tool_strings}
</tools>

Valid tool calls are in the following format:

Tool Calls:
{dumps([{
"id": "call_ID",
"name": "tool_name",
"arguments": {
  "arg1": "value1"
},
"requestor": "assistant"
}], indent=2)}

If the tool calls are not in this format, they are invalid.
""".strip(),
        ),
    )

    openai_messages = to_litellm_messages(review_messages)  # type: ignore
    for message in openai_messages:
        if "tool_calls" in message and message["tool_calls"] is not None:
            message["content"] = "Tool Calls:\n" + dumps(
                message["tool_calls"], indent=2
            )

    openai_response = to_litellm_messages([assistant_message])[0]
    if "tool_calls" in openai_response and openai_response["tool_calls"] is not None:
        openai_response["content"] = dumps(openai_response["tool_calls"], indent=2)

    try:
        trustworthiness = tlm.score(
            messages=openai_messages,
            response={"chat_completion": {"choices": [{"message": openai_response}]}},
            model="gpt-4.1-mini",
        )
        prompt_cost, completion_cost = cost_per_token(
            TLM_MODEL,
            prompt_tokens=trustworthiness["usage"]["prompt_tokens"],
            completion_tokens=trustworthiness["usage"]["completion_tokens"],
        )

        assistant_message.raw_data["trustworthiness"] = trustworthiness
        assistant_message.cost += prompt_cost + completion_cost  # type: ignore
        return trustworthiness

    except Exception as e:
        trustworthiness = {
            "confidence_score": 0.0,
            "explanation": f"Error calculating trustworthiness: {e}",
            "usage": {
                "model": "gpt-4.1-mini",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        assistant_message.raw_data["trustworthiness"] = trustworthiness
        return trustworthiness  # type: ignore


def get_fix_messages(
    assistant_message: AssistantMessage, trustworthiness: InferenceResult
) -> list[APICompatibleMessage]:
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

    fix_request_message = SystemMessage(
        role="system",
        content=f"""Your last response was not trustworthy. Rewrite your response to be more trustworthy.
Feedback on previous response:
{trustworthiness["explanation"]}""",
    )
    return canceled_tool_messages + [fix_request_message]  # type: ignore


def determine_rewrite(
    assistant_message: AssistantMessage,
    new_assistant_message: AssistantMessage,
    rewrite: bool,
) -> AssistantMessage:
    if assistant_message.raw_data is None:
        assistant_message.raw_data = {}
    if new_assistant_message.raw_data is None:
        new_assistant_message.raw_data = {}

    if rewrite:
        new_assistant_message.raw_data["previous_trustworthiness"] = (
            assistant_message.raw_data.get("trustworthiness", {})
        )
        new_assistant_message.raw_data["previous_content"] = assistant_message.content

        new_assistant_message.raw_data["previous_tool_calls"] = (
            assistant_message.tool_calls
        )
        return new_assistant_message
    else:
        assistant_message.raw_data["attempt_trustworthiness"] = (
            new_assistant_message.raw_data.get("trustworthiness", {})
        )
        assistant_message.raw_data["attempt_content"] = new_assistant_message.content
        assistant_message.raw_data["attempt_tool_calls"] = (
            new_assistant_message.tool_calls
        )
        return assistant_message
