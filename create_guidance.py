from json import dumps

from tlm import TLM
from tlm.config.base import ConfigInput

from tau2.utils.guidance import create_guidance, maybe_rewrite_query

tlm = TLM(
    config_input=ConfigInput(
        reasoning_effort="high",  # type: ignore
    )
)

data = {
    "explanation": "e",
    "domain": "airline",
    "response": "",
    "messages": [
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {
            "role": "user",
            "content": "I need to cancel my upcoming flights on reservations **XEHM4B** and **59XX6W**. I’m sick.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_82xO3UdV8FSf5XDBGQJFqsfp",
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": "XEHM4B"},
                    "requestor": "assistant",
                },
                {
                    "id": "call_07j5s25B3Veu1h1lNmg6Igk1",
                    "name": "get_reservation_details",
                    "arguments": {"reservation_id": "59XX6W"},
                    "requestor": "assistant",
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"reservation_id": "XEHM4B", "user_id": "daiki_muller_1116", "origin": "LAS", "destination": "ATL", "flight_type": "round_trip", "cabin": "basic_economy", "flights": [{"flight_number": "HAT005", "origin": "LAS", "destination": "ATL", "date": "2024-05-20", "price": 65}, {"flight_number": "HAT178", "origin": "ATL", "destination": "LAS", "date": "2024-05-30", "price": 83}], "passengers": [{"first_name": "Daiki", "last_name": "Muller", "dob": "1954-07-04"}, {"first_name": "Omar", "last_name": "Ito", "dob": "1975-09-19"}], "payment_history": [{"payment_id": "credit_card_2408938", "amount": 296}], "created_at": "2024-05-01T05:17:41", "total_baggages": 0, "nonfree_baggages": 0, "insurance": "no", "status": null}',
        },
        {
            "role": "tool",
            "content": '{"reservation_id": "59XX6W", "user_id": "daiki_muller_1116", "origin": "LAS", "destination": "ATL", "flight_type": "round_trip", "cabin": "economy", "flights": [{"flight_number": "HAT007", "origin": "LAS", "destination": "ATL", "date": "2024-05-19", "price": 168}, {"flight_number": "HAT174", "origin": "ATL", "destination": "LAS", "date": "2024-05-29", "price": 114}], "passengers": [{"first_name": "Daiki", "last_name": "Muller", "dob": "1954-07-04"}, {"first_name": "Chen", "last_name": "Muller", "dob": "1982-03-25"}], "payment_history": [{"payment_id": "credit_card_2408938", "amount": 624}], "created_at": "2024-05-12T04:19:15", "total_baggages": 0, "nonfree_baggages": 0, "insurance": "yes", "status": null}',
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_AC8L66YnBUVu0RPj4L1OolpL",
                    "name": "get_flight_status",
                    "arguments": {"flight_number": "HAT005", "date": "2024-05-20"},
                    "requestor": "assistant",
                },
                {
                    "id": "call_i606pMFDzvoYUTp7rDAI9ijE",
                    "name": "get_flight_status",
                    "arguments": {"flight_number": "HAT178", "date": "2024-05-30"},
                    "requestor": "assistant",
                },
                {
                    "id": "call_rzoFRVD8mCOC6G1n60ROepCH",
                    "name": "get_flight_status",
                    "arguments": {"flight_number": "HAT007", "date": "2024-05-19"},
                    "requestor": "assistant",
                },
                {
                    "id": "call_AsyETL7mABmxs2QDMlJDrHXb",
                    "name": "get_flight_status",
                    "arguments": {"flight_number": "HAT174", "date": "2024-05-29"},
                    "requestor": "assistant",
                },
            ],
        },
        {"role": "tool", "content": "available"},
        {"role": "tool", "content": "available"},
        {"role": "tool", "content": "available"},
        {"role": "tool", "content": "available"},
        {
            "role": "assistant",
            "content": '{"message":"To cancel reservations **XEHM4B** and **59XX6W**, I first need your **user id** (policy requirement).\\n\\nWhat I can already confirm:\\n- **XEHM4B** (created **2024-05-01 05:17:41**) — **334.7 hours** ago from now (2024-05-15 15:00 EST). Flights **HAT005 (2024-05-20)** and **HAT178 (2024-05-30)** are both **available** (not flown).\\n- **59XX6W** (created **2024-05-12 04:19:15**) — **130.7 hours** ago. Flights **HAT007 (2024-05-19)** and **HAT174 (2024-05-29)** are both **available** (not flown).\\n\\nCancellation eligibility based on what you told me (“I’m sick”):\\n- **XEHM4B**: **Basic economy + no insurance + booked >24 hours ago** → this does **not** meet the cancellation criteria, so I **cannot cancel** it under the policy.\\n- **59XX6W**: **Has travel insurance** → I *can* cancel it for illness.\\n\\nPlease reply with:\\n1) Your **user id** (it should match the reservations’ user), and\\n2) Confirm you want me to cancel **reservation 59XX6W** due to illness (**yes/no**).\\n\\nIf you still need **XEHM4B** canceled, I can transfer you to a human agent to review options."}',
        },
        {
            "role": "user",
            "content": "User id: **daiki_muller_1116**.\n\nFor **XEHM4B**: upgrade it to **business** first (charge the card ending **2135**), then cancel it.\n\nAnd yes—cancel **59XX6W** due to illness.",
        },
    ],
}

data["messages"] = [
    {"role": m["role"], "content": m.get("content", dumps(m.get("tool_calls", [])))}
    for m in data["messages"]
]

query = [m for m in data["messages"] if m["role"] == "user"][-1]["content"]

print(
    maybe_rewrite_query(
        query=query,
        rewritten_question=None,
        messages=data["messages"],
    )
)

# create_guidance(
#     tlm=tlm,
#     query=query,
#     messages=data["messages"],  # type: ignore
#     explanation=data["explanation"],
#     response=data["response"],
#     domain=data["domain"],
# )
