from asyncio import gather, get_event_loop

import re
from typing import cast
from litellm import acompletion
from openai.types.chat import ChatCompletionMessageParam
from cleanlab_tlm.utils.chat import form_prompt_string
from json import dump, dumps, load

from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.utils.llm_utils import generate, to_litellm_messages

all_guidance: list[dict] = None  # type: ignore

MAX_MESSAGES_TO_CONSIDER = 35
REWRITE_QUERY_PROMPT = """You are a query rewriter system to formulate a self-contained query out of a user message that may (implicitly) reference prior message history.

You will be given a Message History, the Latest User Message, and a Question that will be asked of the Latest User Message.
Your output should be a rewritten query that enables anyone to understand what the user is requesting, without having to know the user's Message History or the Latest User Message.
It must also have enough information for the Question to be answerable.

Notes:
- If the Latest User Message is self-contained, then output it verbatim.
- If you decide to rewrite the Latest User Message to form a more understandable query, then incorporate Message History in your rewritten query ONLY when you can confidently resolve references.
- It is preferable to output the Latest User Message verbatim—even if not fully self-contained—rather than risk misrepresenting it.

Decision gate:
1) If the Latest User Message is conversational (non-information-seeking or phatic statement), then return the original VERBATIM.
2) If the Latest User Message is a self-contained question or request, then return the original VERBATIM.
3) If the Latest User Message contains deixis/possessives (this/that/it/they/these/those/here/there, “the other ...”, his/her/their/he/she/they, then/when) AND those can be uniquely resolved from the prior Message History, then rewrite the User Message following the given Rewriting Rules.
4) If the Latest User Message needs prior Message History as context AND you can uniquely resolve what is being referenced, then rewrite the User Message following the given Rewriting Rules.
5) Otherwise, return the Latest User Message VERBATIM.

Rewriting Rules:
- Do not alter the Latest User Message more than is necessary to make it understandable in isolation.
- Do not add a lot of extra information, anything you add should be concise.
- Preserve the intent of the Latest User Message.
- Resolve references ONLY with facts/information explicitly stated in the Message History. Never invent or guess.
- Replace vague words with the exact resolved noun phrase(s) from the Message History.
- Replace vague intent with the exact resolved intent from Message History (example: "What are other AI tools with similar branding?" -> "What are other AI tools with branding similar to ChatGPT?")
- Keep appropriate specificity: include the key entity/topic if clear from Message History; omit incidental details from Message History that are not essential to understanding the Latest User Message.
- Impersonal phrasing: avoid “you/your/I/my/we/our” outside quotes.
- Convert assistant-directed forms ("Can you...", "Could you...", "Would you..."") into general-question form ONLY IF the Latest User Message is a question. Do NOT convert imperatives to questions.
- Do NOT append examples, steps, or lists. Never add clauses starting with “including”, “such as”, “like”, or “for example”.

## Inputs to Query Rewriter

<message_history>
{messages}
</message_history>

<latest_user_message>
{query}
</latest_user_message>

<question>
{question}
</question>

## Output Instructions

Do NOT answer the User Message, only rewrite it.

Output a self-contained version of the latest user message, that is minimally modified to account for their prior message history if necessary.

Keep the above rules in mind. Output only your rewritten version of the latest user message, nothing else."""
GUIDANCE_DETERMINATION_PROMPT = """You are an expert assistant that determines whether a certain question is true about the query you are provided. You are only to respond with the word "Yes" or "No" and ABSOLUTELY NOTHING ELSE."""
AI_GUIDANCE_PROMPT = """
Your role is to rephrase Review statements into Guidance statements when necessary.

Below you are given a Query and AI-generated Response to that Query, along with an expert's Review of the Response.
Your task will be to consider whether the Review should be rephrased to be effective Guidance, and if so rewrite it.

<query>
{query}
</query>

<response>
{response}
</response>

<review>
{review}
</review>

## Instructions

First, determine whether the Review needs to be rewritten, in order for it to be an effective Guidance statement (see criteria below).
If the Review does not seem like an effective Guidance statement, then rewrite it to be one.

### Criteria for an effective Guidance statement

The Guidance will be added into the AI's system prompt (in the appropriate place), to help the AI generate better responses for queries similar to this one. It should be a concise self-contained statement/paragraph that assumes the overall task is already known to the AI.

An effective Guidance statement should follow this template:

> If [scenario], then [advice]

The *scenario* should be a short description of the types of query this Guidance is intended for.
The *advice* should be phrased as tips on how to respond in this scenario (use phrases like "you should" to directly address the AI).
The advice need NOT include explanations for why it is being given.


It is possible that the Guidance will be given to the AI in scenarios where it shouldn't have been, which is why it is important that the Guidance start with an "If <scenario>" statement to clarify when it is relevant.

The Guidance should not be overly focused on pointing out particular flaws in this specific Response (i.e. phrase it as advice rather than feedback).
For example: "If the user is simply saying goodbye or thank you, then respond in under 10 words." is better guidance than: "This response was too long, since the user was just saying goodbye, it should be shortened to under 10 words".
The Guidance will be given to the AI, without the AI having seen this specific Response.
If response-specific feedback seems critical to include, then templates like this can be used for the Guidance:
> If [scenario], then do [advice] instead of [feedback phrased as general things to avoid].
> If [scenario], then do NOT [feedback phrased as general things to avoid]. Instead [advice].

Examples of effective Guidance statements from other use-cases:
- "If the customer is asking about their account, include this link ([link1]) in your response instead of this link ([link2])."
- "If the user seems angry, then apologize, respond empathetically, and let them know: they can share their feedback by emailing support@acme.com and our team will look into what happened here."

Good guidance must be concise.


### Guidelines for rewriting a Review

Rewrite things very concisely, while ensuring your Guidance is effective according to the above criteria.

When rewriting, assume the Review comes from an authoritative expert and keep your Guidance meticulously faithful to the Review.
Assume the expert knows significantly more information than the AI receiving your Guidance.

Do NOT add extra motivations/reasons for why the Guidance should be heeded.
Do NOT make assumptions about why the expert gave their Review.
Do NOT infer ANY extra instructions/desiderata beyond what can be unequivocally concluded from the Review.


## Output Format

If the Review already seems like an effective Guidance statement, then simply output it verbatim.
Otherwise, rewrite the Review into an effective Guidance statement.
"""


def format_messages_to_string(
    messages: list[ChatCompletionMessageParam], query: str
) -> str:
    """Format messages for inclusion in a prompt. Assumes messages has at least one message."""
    formatted_messages: list[ChatCompletionMessageParam] = []

    if len(messages) > 0:
        messages = messages[-MAX_MESSAGES_TO_CONSIDER:]
        if messages[-1]["role"] == "user":
            messages = messages[
                :-1
            ]  # Remove the last user message (user prompt in some cases) to replace with simply the query

        for message in messages:
            if message["role"] == "user":  # Dictionary access
                formatted_messages.append(
                    {"role": "user", "content": message["content"]}
                )
            elif message["role"] == "assistant":  # Dictionary access
                content = ""
                if isinstance(message["content"], str):  # type: ignore
                    content = message["content"]  # type: ignore
                elif isinstance(message["content"], list):  # type: ignore
                    content = "\n".join(
                        [
                            m.get("text", "")
                            for m in message["content"]  # type: ignore
                            if isinstance(m, dict) and "text" in m
                        ]
                    )

                # Only include assistant messages that have actual content (skip tool_calls without content)
                if content:
                    formatted_messages.append({"role": "assistant", "content": content})
            # Skip tool messages - they contain internal implementation details that shouldn't be in the rewrite context

    formatted_messages.append({"role": "user", "content": query})
    messages_str = form_prompt_string(formatted_messages)  # type: ignore
    # Remove 'Assistant: ' prefix if present (added to the end of string by form_prompt_string)
    trailing_assistant_prefix_pattern = rf"\s*Assistant:\s*$"
    messages_str = re.sub(trailing_assistant_prefix_pattern, "", messages_str)
    return messages_str


async def maybe_rewrite_query(
    query: str,
    question: str,
    messages: list[ChatCompletionMessageParam] | None,
) -> str:
    """Handles the logic for rewriting a query if needed.
    - If messages are provided and the conversation is longer than a single turn it will call the LLM to rewrite the query.
    """

    if messages is None or len(messages) == 0:
        return query

    messages_str = format_messages_to_string(messages, query)

    response = await acompletion(
        model="gpt-5-mini",
        messages=[
            {
                "role": "user",
                "content": REWRITE_QUERY_PROMPT.format(
                    query=query, messages=messages_str, question=question
                ),
            }
        ],
    )

    if response is not None:
        return response.choices[0].message.content or ""  # type: ignore

    return query  # Return original query if response is None


# def create_guidance(
#     query: str,
#     messages: list[ChatCompletionMessageParam],
#     explanation: str,
#     response: str,
#     tlm: TLM,
#     domain: str,
# ):
#     global all_guidance
#     GUIDANCE_FILE = f"data/tau2/domains/{domain}/guidance.json"

#     try:
#         with open(GUIDANCE_FILE, "r") as f:
#             all_guidance = load(f)
#     except FileNotFoundError:
#         all_guidance = []

#     new_query = maybe_rewrite_query(query, None, messages)

#     guidance = tlm.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": AI_GUIDANCE_PROMPT.format(
#                     query=query, review=explanation, response=response
#                 ),
#             }
#         ],
#     )

#     to_add = {"query": new_query, "guidance": guidance["response"]["choices"][0]["message"]["content"]}  # type: ignore

#     all_guidance.append(to_add)

#     with open(GUIDANCE_FILE, "w") as f:
#         dump(all_guidance, f, indent=2)


def consult_ai_guidance(
    messages: list[ChatCompletionMessageParam],
    triggers="before",
) -> list[str]:

    async def determine_guidance_relevance(guidance: dict) -> bool:
        if messages[-1]["role"] == "tool":
            if guidance["type"] != "tool":
                return False
            rewritten_query = ""
            for message in messages[::-1]:
                if message["role"] == "tool":
                    content = cast(str, message["content"])
                    rewritten_query = "Tool:\n" + content + "\n" + rewritten_query
                else:
                    tool_calls = message["tool_calls"]  # type: ignore
                    rewritten_query = (
                        f"{message['role'].capitalize()}:\n{dumps(tool_calls, indent=2)}\n"
                        + rewritten_query
                    )
                    break
        else:
            if guidance["type"] == "tool":
                return False
            rewritten_query = await maybe_rewrite_query(
                messages[-1].content or "", guidance["query"], messages  # type: ignore
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
                    "content": f"""Query: {rewritten_query}
Question: {guidance["query"]}""",
                },
            ],
        )

        return (
            "yes" in determination_response.choices[0].message.content.strip().lower()  # type: ignore
        )

    print([guidance for guidance in all_guidance if guidance["triggers"] == triggers])

    request = get_event_loop().run_until_complete(
        gather(
            *[
                determine_guidance_relevance(guidance)
                for guidance in all_guidance
                if guidance["triggers"] == triggers
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
    }

    guidance = consult_ai_guidance(to_litellm_messages(messages))  # type: ignore

    total_guidance = set(guidance) | previous_guidance

    return guidance, (
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
    }

    guidance = consult_ai_guidance(to_litellm_messages(messages), triggers="after")  # type: ignore

    if not guidance:
        return [], []

    return guidance, (
        [
            SystemMessage(
                role="system",
                content=(
                    f"""There may have been some issues with your previous message. You must rewrite it to address the following guidance. You must follow the rules that apply. If they don't apply, you can ignore them. If none apply or no changes are needed, just repeat your previous message verbatim:
{"\n".join([f"- {g}" for g in guidance])}"""
                    + f"""

Be sure that if you rewrite your message, you still adhere to the following guidance as well. If they don't apply, you can ignore them:
{"\n".join([f"- {g}" for g in previous_guidance])}"""
                    if previous_guidance
                    else ""
                ),
            )
        ]
    )
