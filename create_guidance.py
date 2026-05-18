from json import dumps

from tlm import TLM
from tlm.config.base import ConfigInput

from tau2.utils.guidance import create_guidance

tlm = TLM(
    config_input=ConfigInput(
        reasoning_effort="high",  # type: ignore
    )
)

data = {
    "explanation": "If the user asks for a refund on a cancelation, ensure their reservation was made less than 24 hours ago by checking the date and especially the time.",
    "domain": "airline",
    "response": '[\n  {\n    "id": "call_4CpTja9DkjUFKQvpJcORdlee",\n    "name": "get_user_details",\n    "arguments": {\n      "user_id": "raj_sanchez_7340"\n    },\n    "requestor": "assistant"\n  }\n]',
    "messages": [
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {
            "role": "user",
            "content": "Hi—I'm Raj Sanchez (user id: raj_sanchez_7340). I was told by phone support that a service agent could help me cancel my reservation for the trip from Philadelphia to LaGuardia. I’d like to cancel it, but only if I’m getting a refund.",
        },
    ],
}

data["messages"] = [
    {"role": m["role"], "content": m.get("content", dumps(m.get("tool_calls", [])))}
    for m in data["messages"]
]

query = [m for m in data["messages"] if m["role"] == "user"][-1]["content"]

create_guidance(
    tlm=tlm,
    query=query,
    messages=data["messages"],  # type: ignore
    explanation=data["explanation"],
    response=data["response"],
    domain=data["domain"],
)
