from json import loads

from litellm import completion
from tlm import TLM

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.domains.airline.data_model import FlightDB, get_db
from tau2.domains.airline.tools import AirlineTools
from tau2.utils.llm_utils import generate, to_tau2_messages
from tau2.utils.trustworthiness import trustworthiness_from_messages

null = None
false = False
true = True

db: FlightDB = get_db()  # type: ignore
tools = list(AirlineTools(db).get_tools().values())

with open("data/tau2/domains/airline/policy.md", "r") as f:
    domain_policy = f.read()

system = {
    "role": "system",
    "content": SYSTEM_PROMPT.format(
        domain_policy=domain_policy, agent_instruction=AGENT_INSTRUCTION
    ),
}

messages = [
    {
        "role": "assistant",
        "content": "Hi! How can I help you today?",
        "tool_calls": null,
        "is_audio": false,
        "turn_idx": 0,
        "timestamp": "2026-05-24T16:35:03.189224",
        "cost": 0,
        "usage": null,
        "raw_data": null,
        "generation_time_seconds": null,
        "audio_format": null,
        "audio_path": null,
        "audio_script_gold": null,
        "speech_effects": null,
        "source_effects": null,
        "channel_effects": null,
        "turn_taking_action": null,
        "utterance_ids": null,
        "chunk_id": null,
        "is_final_chunk": true,
        "source": null,
        "contains_speech": true,
    },
    {
        "role": "user",
        "content": "Hi—I'm the customer. I need help with two things:\n\n1) I want to remove a passenger (Ethan) from my reservation **H9ZU1C**. If that isn’t possible, please cancel the reservation and I’ll rebook later.\n\n2) I also want to book the **cheapest direct round-trip** flight from **New York (JFK or EWR)** to **any West Coast** destination, departing **May 20** and returning **May 25**. I’m fine with **basic economy** if it’s cheaper. I want to **use my smaller gift card first, then the larger one**, use **all free baggage allowance**, and **no insurance**. My DOB is in my profile—please look it up.\n\nMy name is **Mia Kim**, user id **mia_kim_4397**.",
        "tool_calls": null,
        "is_audio": false,
        "turn_idx": 1,
        "timestamp": "2026-05-24T16:35:08.441525",
        "cost": 0.003556,
        "usage": {"completion_tokens": 178, "prompt_tokens": 608},
        "raw_data": {
            "id": "chatcmpl-Dj9pKhyM7dPzmRX2xBxOif2dtzMu9",
            "created": 1779654906,
            "model": "gpt-5.2-2025-12-11",
            "object": "chat.completion",
            "system_fingerprint": null,
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Hi—I'm the customer. I need help with two things:\n\n1) I want to remove a passenger (Ethan) from my reservation **H9ZU1C**. If that isn’t possible, please cancel the reservation and I’ll rebook later.\n\n2) I also want to book the **cheapest direct round-trip** flight from **New York (JFK or EWR)** to **any West Coast** destination, departing **May 20** and returning **May 25**. I’m fine with **basic economy** if it’s cheaper. I want to **use my smaller gift card first, then the larger one**, use **all free baggage allowance**, and **no insurance**. My DOB is in my profile—please look it up.\n\nMy name is **Mia Kim**, user id **mia_kim_4397**.",
                        "role": "assistant",
                        "tool_calls": null,
                        "function_call": null,
                        "provider_specific_fields": {"refusal": null},
                        "annotations": [],
                    },
                    "provider_specific_fields": {},
                }
            ],
            "usage": {
                "completion_tokens": 178,
                "prompt_tokens": 608,
                "total_tokens": 786,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            },
            "service_tier": "default",
        },
        "generation_time_seconds": null,
        "audio_format": null,
        "audio_path": null,
        "audio_script_gold": null,
        "speech_effects": null,
        "source_effects": null,
        "channel_effects": null,
        "turn_taking_action": null,
        "utterance_ids": null,
        "chunk_id": null,
        "is_final_chunk": true,
        "source": null,
        "contains_speech": true,
    },
    {
        "role": "assistant",
        "content": null,
        "tool_calls": [
            {
                "id": "call_LBQxLvMQs2pVpDG51xZm9yMt",
                "name": "get_user_details",
                "arguments": {"user_id": "mia_kim_4397"},
                "requestor": "assistant",
            }
        ],
        "is_audio": false,
        "turn_idx": 2,
        "timestamp": "2026-05-24T16:35:11.932359",
        "cost": 0.006699,
        "usage": {"completion_tokens": 24, "prompt_tokens": 3636},
        "raw_data": {
            "id": "chatcmpl-Dj9pPyS39BPYgxu6ik747Ys57kNiK",
            "created": 1779654911,
            "model": "gpt-5.2-2025-12-11",
            "object": "chat.completion",
            "system_fingerprint": null,
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "content": null,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": '{"user_id":"mia_kim_4397"}',
                                    "name": "get_user_details",
                                },
                                "id": "call_LBQxLvMQs2pVpDG51xZm9yMt",
                                "type": "function",
                            }
                        ],
                        "function_call": null,
                        "provider_specific_fields": {"refusal": null},
                        "annotations": [],
                    },
                    "provider_specific_fields": {},
                }
            ],
            "usage": {
                "completion_tokens": 24,
                "prompt_tokens": 3636,
                "total_tokens": 3660,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            },
            "service_tier": "default",
            "guidance": [
                "If the user wants to cancel a reservation, state the exact number of hours between the current date and the booking date. Also search each flight number to ensure none have been flown. List the specific cancellation criteria in the policy, and if the reservation fails to meet any of them, refuse the cancellation. You are allowed to make the necessary changes to a reservation to make it cancellable, including upgrading to business."
            ],
        },
        "generation_time_seconds": 0.9939125500386581,
        "audio_format": null,
        "audio_path": null,
        "audio_script_gold": null,
        "speech_effects": null,
        "source_effects": null,
        "channel_effects": null,
        "turn_taking_action": null,
        "utterance_ids": null,
        "chunk_id": null,
        "is_final_chunk": true,
        "source": null,
        "contains_speech": true,
    },
    {
        "id": "call_LBQxLvMQs2pVpDG51xZm9yMt",
        "role": "tool",
        "content": '{"user_id": "mia_kim_4397", "name": {"first_name": "Mia", "last_name": "Kim"}, "address": {"address1": "681 Cedar Street", "address2": "Suite 847", "city": "Washington", "country": "USA", "state": "DC", "zip": "20017"}, "email": "mia.kim6850@example.com", "dob": "1965-06-09", "payment_methods": {"gift_card_7359776": {"source": "gift_card", "id": "gift_card_7359776", "amount": 39.0}, "gift_card_7773485": {"source": "gift_card", "id": "gift_card_7773485", "amount": 203.0}}, "saved_passengers": [{"first_name": "Ethan", "last_name": "Lopez", "dob": "1991-11-04"}], "membership": "silver", "reservations": ["C4GP7M", "XL2C75", "H9ZU1C"]}',
        "requestor": "assistant",
        "error": false,
        "turn_idx": 3,
        "timestamp": "2026-05-24T16:35:11.932550",
    },
    {
        "role": "assistant",
        "content": null,
        "tool_calls": [
            {
                "id": "call_AbFpRA2h5usI7tOj4mtIUbsf",
                "name": "get_reservation_details",
                "arguments": {"reservation_id": "H9ZU1C"},
                "requestor": "assistant",
            }
        ],
        "is_audio": false,
        "turn_idx": 4,
        "timestamp": "2026-05-24T16:35:14.460353",
        "cost": 0.00213325,
        "usage": {"completion_tokens": 24, "prompt_tokens": 3907},
        "raw_data": {
            "id": "chatcmpl-Dj9pRDRXdRC8IkxHWNCbYSf4mMOuq",
            "created": 1779654913,
            "model": "gpt-5.2-2025-12-11",
            "object": "chat.completion",
            "system_fingerprint": null,
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "content": null,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": '{"reservation_id":"H9ZU1C"}',
                                    "name": "get_reservation_details",
                                },
                                "id": "call_AbFpRA2h5usI7tOj4mtIUbsf",
                                "type": "function",
                            }
                        ],
                        "function_call": null,
                        "provider_specific_fields": {"refusal": null},
                        "annotations": [],
                    },
                    "provider_specific_fields": {},
                }
            ],
            "usage": {
                "completion_tokens": 24,
                "prompt_tokens": 3907,
                "total_tokens": 3931,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 3200},
            },
            "service_tier": "default",
            "guidance": [],
        },
        "generation_time_seconds": 1.0781009860220365,
        "audio_format": null,
        "audio_path": null,
        "audio_script_gold": null,
        "speech_effects": null,
        "source_effects": null,
        "channel_effects": null,
        "turn_taking_action": null,
        "utterance_ids": null,
        "chunk_id": null,
        "is_final_chunk": true,
        "source": null,
        "contains_speech": true,
    },
    {
        "id": "call_AbFpRA2h5usI7tOj4mtIUbsf",
        "role": "tool",
        "content": '{"reservation_id": "H9ZU1C", "user_id": "mia_kim_4397", "origin": "MIA", "destination": "IAH", "flight_type": "one_way", "cabin": "economy", "flights": [{"flight_number": "HAT192", "origin": "MIA", "destination": "EWR", "date": "2024-05-24", "price": 174}, {"flight_number": "HAT166", "origin": "EWR", "destination": "IAH", "date": "2024-05-25", "price": 125}], "passengers": [{"first_name": "James", "last_name": "Moore", "dob": "1977-02-03"}, {"first_name": "Ethan", "last_name": "Lopez", "dob": "1991-11-04"}, {"first_name": "Amelia", "last_name": "Sanchez", "dob": "1968-08-24"}], "payment_history": [{"payment_id": "gift_card_7359776", "amount": 897}], "created_at": "2024-05-01T04:50:18", "total_baggages": 0, "nonfree_baggages": 0, "insurance": "no", "status": null}',
        "requestor": "assistant",
        "error": false,
        "turn_idx": 5,
        "timestamp": "2026-05-24T16:35:14.460505",
    },
    {
        "role": "assistant",
        "content": null,
        "tool_calls": [
            {
                "id": "call_QHlkLYOvot7JIfL3NoVTEHnF",
                "name": "get_flight_status",
                "arguments": {"flight_number": "HAT192", "date": "2024-05-24"},
                "requestor": "assistant",
            },
            {
                "id": "call_ONezNJ7s29O25Iz4eLqAci35",
                "name": "get_flight_status",
                "arguments": {"flight_number": "HAT166", "date": "2024-05-25"},
                "requestor": "assistant",
            },
            {
                "id": "call_Ffa3y6FZQyidAztyJTuiOXRe",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "JFK",
                    "destination": "LAX",
                    "date": "2024-05-20",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_JuRY4CKXh2jb7QLUYlnDgqGz",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "JFK",
                    "destination": "SFO",
                    "date": "2024-05-20",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_l6jqd57JJH2HW3wmwx2cb0Zm",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "JFK",
                    "destination": "SEA",
                    "date": "2024-05-20",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_LnlcjYnXGSB3iBfb6oCFgWGP",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "JFK",
                    "destination": "SAN",
                    "date": "2024-05-20",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_FjA3ZK2rklqDxSCua974XbvJ",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "JFK",
                    "destination": "LAS",
                    "date": "2024-05-20",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_eLc7w3u0q6w6CyrEqRn7J5dS",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "EWR",
                    "destination": "LAX",
                    "date": "2024-05-20",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_vd30FZGGEoQPTvocOoq30uzD",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "EWR",
                    "destination": "SFO",
                    "date": "2024-05-20",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_Ji54CJBSAHDCpKyCXIRYucXw",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "EWR",
                    "destination": "SEA",
                    "date": "2024-05-20",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_mBx6YwbUBrkzCpaqa90Md3hI",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "EWR",
                    "destination": "SAN",
                    "date": "2024-05-20",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_FckJ3ZrwMjLHtPIi7rFJUwYv",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "EWR",
                    "destination": "LAS",
                    "date": "2024-05-20",
                },
                "requestor": "assistant",
            },
        ],
        "is_audio": false,
        "turn_idx": 6,
        "timestamp": "2026-05-24T16:35:18.434070",
        "cost": 0.0068743499999999996,
        "usage": {"completion_tokens": 379, "prompt_tokens": 4237},
        "raw_data": {
            "id": "chatcmpl-Dj9pSzL9xiFoJRt2OyYw45iPeZejv",
            "created": 1779654914,
            "model": "gpt-5.2-2025-12-11",
            "object": "chat.completion",
            "system_fingerprint": null,
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "content": null,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": '{"flight_number": "HAT192", "date": "2024-05-24"}',
                                    "name": "get_flight_status",
                                },
                                "id": "call_QHlkLYOvot7JIfL3NoVTEHnF",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"flight_number": "HAT166", "date": "2024-05-25"}',
                                    "name": "get_flight_status",
                                },
                                "id": "call_ONezNJ7s29O25Iz4eLqAci35",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "JFK", "destination": "LAX", "date": "2024-05-20"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_Ffa3y6FZQyidAztyJTuiOXRe",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "JFK", "destination": "SFO", "date": "2024-05-20"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_JuRY4CKXh2jb7QLUYlnDgqGz",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_l6jqd57JJH2HW3wmwx2cb0Zm",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "JFK", "destination": "SAN", "date": "2024-05-20"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_LnlcjYnXGSB3iBfb6oCFgWGP",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "JFK", "destination": "LAS", "date": "2024-05-20"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_FjA3ZK2rklqDxSCua974XbvJ",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "EWR", "destination": "LAX", "date": "2024-05-20"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_eLc7w3u0q6w6CyrEqRn7J5dS",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "EWR", "destination": "SFO", "date": "2024-05-20"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_vd30FZGGEoQPTvocOoq30uzD",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "EWR", "destination": "SEA", "date": "2024-05-20"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_Ji54CJBSAHDCpKyCXIRYucXw",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "EWR", "destination": "SAN", "date": "2024-05-20"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_mBx6YwbUBrkzCpaqa90Md3hI",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "EWR", "destination": "LAS", "date": "2024-05-20"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_FckJ3ZrwMjLHtPIi7rFJUwYv",
                                "type": "function",
                            },
                        ],
                        "function_call": null,
                        "provider_specific_fields": {"refusal": null},
                        "annotations": [],
                    },
                    "provider_specific_fields": {},
                }
            ],
            "usage": {
                "completion_tokens": 379,
                "prompt_tokens": 4237,
                "total_tokens": 4616,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 3712},
            },
            "service_tier": "default",
            "guidance": [],
        },
        "generation_time_seconds": 3.6389818849856965,
        "audio_format": null,
        "audio_path": null,
        "audio_script_gold": null,
        "speech_effects": null,
        "source_effects": null,
        "channel_effects": null,
        "turn_taking_action": null,
        "utterance_ids": null,
        "chunk_id": null,
        "is_final_chunk": true,
        "source": null,
        "contains_speech": true,
    },
    {
        "id": "call_QHlkLYOvot7JIfL3NoVTEHnF",
        "role": "tool",
        "content": "available",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 7,
        "timestamp": "2026-05-24T16:35:18.434534",
    },
    {
        "id": "call_ONezNJ7s29O25Iz4eLqAci35",
        "role": "tool",
        "content": "available",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 8,
        "timestamp": "2026-05-24T16:35:18.434552",
    },
    {
        "id": "call_Ffa3y6FZQyidAztyJTuiOXRe",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 9,
        "timestamp": "2026-05-24T16:35:18.434621",
    },
    {
        "id": "call_JuRY4CKXh2jb7QLUYlnDgqGz",
        "role": "tool",
        "content": '[{"flight_number": "HAT023", "origin": "JFK", "destination": "SFO", "status": "available", "scheduled_departure_time_est": "14:00:00", "scheduled_arrival_time_est": "20:00:00", "date": null, "available_seats": {"basic_economy": 20, "economy": 13, "business": 11}, "prices": {"basic_economy": 100, "economy": 191, "business": 473}}]',
        "requestor": "assistant",
        "error": false,
        "turn_idx": 10,
        "timestamp": "2026-05-24T16:35:18.434707",
    },
    {
        "id": "call_l6jqd57JJH2HW3wmwx2cb0Zm",
        "role": "tool",
        "content": '[{"flight_number": "HAT069", "origin": "JFK", "destination": "SEA", "status": "available", "scheduled_departure_time_est": "06:00:00", "scheduled_arrival_time_est": "12:00:00", "date": null, "available_seats": {"basic_economy": 17, "economy": 12, "business": 3}, "prices": {"basic_economy": 51, "economy": 121, "business": 239}}, {"flight_number": "HAT083", "origin": "JFK", "destination": "SEA", "status": "available", "scheduled_departure_time_est": "01:00:00", "scheduled_arrival_time_est": "07:00:00", "date": null, "available_seats": {"basic_economy": 16, "economy": 7, "business": 3}, "prices": {"basic_economy": 87, "economy": 100, "business": 276}}]',
        "requestor": "assistant",
        "error": false,
        "turn_idx": 11,
        "timestamp": "2026-05-24T16:35:18.434774",
    },
    {
        "id": "call_LnlcjYnXGSB3iBfb6oCFgWGP",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 12,
        "timestamp": "2026-05-24T16:35:18.434800",
    },
    {
        "id": "call_FjA3ZK2rklqDxSCua974XbvJ",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 13,
        "timestamp": "2026-05-24T16:35:18.434822",
    },
    {
        "id": "call_eLc7w3u0q6w6CyrEqRn7J5dS",
        "role": "tool",
        "content": '[{"flight_number": "HAT041", "origin": "EWR", "destination": "LAX", "status": "available", "scheduled_departure_time_est": "07:00:00", "scheduled_arrival_time_est": "12:30:00", "date": null, "available_seats": {"basic_economy": 9, "economy": 10, "business": 16}, "prices": {"basic_economy": 86, "economy": 196, "business": 306}}]',
        "requestor": "assistant",
        "error": false,
        "turn_idx": 14,
        "timestamp": "2026-05-24T16:35:18.434860",
    },
    {
        "id": "call_vd30FZGGEoQPTvocOoq30uzD",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 15,
        "timestamp": "2026-05-24T16:35:18.434882",
    },
    {
        "id": "call_Ji54CJBSAHDCpKyCXIRYucXw",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 16,
        "timestamp": "2026-05-24T16:35:18.434903",
    },
    {
        "id": "call_mBx6YwbUBrkzCpaqa90Md3hI",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 17,
        "timestamp": "2026-05-24T16:35:18.434924",
    },
    {
        "id": "call_FckJ3ZrwMjLHtPIi7rFJUwYv",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 18,
        "timestamp": "2026-05-24T16:35:18.434943",
    },
    {
        "role": "assistant",
        "content": "1) Removing passenger “Ethan” from reservation **H9ZU1C**\n- This is **not possible**: passenger changes are allowed, but **the number of passengers cannot be changed** (so I can’t remove Ethan).\n\nIf you want, I *can* cancel reservation **H9ZU1C** instead, but I need to confirm it’s eligible:\n\n**Flight status check (to ensure none have been flown):**\n- **HAT192 (2024-05-24):** available (not flown)\n- **HAT166 (2024-05-25):** available (not flown)\n\n**Time since booking (required):**\n- Current time: **2024-05-15 15:00 EST**\n- Reservation created: **2024-05-01 04:50:18**\n- Time between: **348.16 hours** (approx.)\n\n**Cancellation criteria (must meet at least one):**\n- Booking was made within the last 24 hrs (**No**)\n- Flight is cancelled by airline (**Not indicated**)\n- It is a business flight (**No — cabin is economy**)\n- User has travel insurance and reason is covered (**No — insurance is “no”**)\n\nBased on the above, this reservation **does not meet any cancellation criteria**, so I **can’t cancel H9ZU1C**.\n\nIf you still want to proceed, I can **transfer you to a human agent** to review options.\n\n---\n\n2) Booking the cheapest direct round-trip (May 20 → May 25), NY (JFK/EWR) → West Coast, basic economy, no insurance\n\nI can search and book, but I need two things first:\n1) **How many passengers** is this new booking for? (Just you, or multiple?)\n2) For the return on **May 25**, which West Coast airports should I consider? I can currently search direct flights to: **LAX, SFO, SEA, SAN, LAS** (tell me which are acceptable).\n\nWhat I found so far for **departing May 20 (direct)**:\n- **JFK → SEA** flight **HAT069**: **Basic Economy $51** (cheapest found)\n- **EWR → LAX** flight **HAT041**: **Basic Economy $86**\n- **JFK → SFO** flight **HAT023**: **Basic Economy $100**\n(Other searched routes returned no direct options for that date.)\n\nPayment/baggage notes based on your profile:\n- You have gift cards: **$39 (gift_card_7359776)** and **$203 (gift_card_7773485)** — I can apply the **smaller first**, then the larger.\n- You are **Silver**. Free checked bag allowance depends on cabin; once you confirm passenger count and we pick flights, I’ll calculate the **maximum free bags** and only add up to that (no paid bags unless you ask).\n- Insurance will be set to **no**.\n- Your DOB in profile is **1965-06-09** (for you).\n\nReply with:\n- passenger count, and\n- acceptable West Coast destination airports for the return search (LAX/SFO/SEA/SAN/LAS).",
        "tool_calls": null,
        "is_audio": false,
        "turn_idx": 19,
        "timestamp": "2026-05-24T16:35:27.839208",
        "cost": 0.0127162,
        "usage": {"completion_tokens": 674, "prompt_tokens": 5100},
        "raw_data": {
            "id": "chatcmpl-Dj9pXgDJHgpj6e3495YokdFES4sm2",
            "created": 1779654919,
            "model": "gpt-5.2-2025-12-11",
            "object": "chat.completion",
            "system_fingerprint": null,
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "1) Removing passenger “Ethan” from reservation **H9ZU1C**\n- This is **not possible**: passenger changes are allowed, but **the number of passengers cannot be changed** (so I can’t remove Ethan).\n\nIf you want, I *can* cancel reservation **H9ZU1C** instead, but I need to confirm it’s eligible:\n\n**Flight status check (to ensure none have been flown):**\n- **HAT192 (2024-05-24):** available (not flown)\n- **HAT166 (2024-05-25):** available (not flown)\n\n**Time since booking (required):**\n- Current time: **2024-05-15 15:00 EST**\n- Reservation created: **2024-05-01 04:50:18**\n- Time between: **348.16 hours** (approx.)\n\n**Cancellation criteria (must meet at least one):**\n- Booking was made within the last 24 hrs (**No**)\n- Flight is cancelled by airline (**Not indicated**)\n- It is a business flight (**No — cabin is economy**)\n- User has travel insurance and reason is covered (**No — insurance is “no”**)\n\nBased on the above, this reservation **does not meet any cancellation criteria**, so I **can’t cancel H9ZU1C**.\n\nIf you still want to proceed, I can **transfer you to a human agent** to review options.\n\n---\n\n2) Booking the cheapest direct round-trip (May 20 → May 25), NY (JFK/EWR) → West Coast, basic economy, no insurance\n\nI can search and book, but I need two things first:\n1) **How many passengers** is this new booking for? (Just you, or multiple?)\n2) For the return on **May 25**, which West Coast airports should I consider? I can currently search direct flights to: **LAX, SFO, SEA, SAN, LAS** (tell me which are acceptable).\n\nWhat I found so far for **departing May 20 (direct)**:\n- **JFK → SEA** flight **HAT069**: **Basic Economy $51** (cheapest found)\n- **EWR → LAX** flight **HAT041**: **Basic Economy $86**\n- **JFK → SFO** flight **HAT023**: **Basic Economy $100**\n(Other searched routes returned no direct options for that date.)\n\nPayment/baggage notes based on your profile:\n- You have gift cards: **$39 (gift_card_7359776)** and **$203 (gift_card_7773485)** — I can apply the **smaller first**, then the larger.\n- You are **Silver**. Free checked bag allowance depends on cabin; once you confirm passenger count and we pick flights, I’ll calculate the **maximum free bags** and only add up to that (no paid bags unless you ask).\n- Insurance will be set to **no**.\n- Your DOB in profile is **1965-06-09** (for you).\n\nReply with:\n- passenger count, and\n- acceptable West Coast destination airports for the return search (LAX/SFO/SEA/SAN/LAS).",
                        "role": "assistant",
                        "tool_calls": null,
                        "function_call": null,
                        "provider_specific_fields": {"refusal": null},
                        "annotations": [],
                    },
                    "provider_specific_fields": {},
                }
            ],
            "usage": {
                "completion_tokens": 674,
                "prompt_tokens": 5100,
                "total_tokens": 5774,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 3584},
            },
            "service_tier": "default",
            "guidance": [],
        },
        "generation_time_seconds": 8.913084010011517,
        "audio_format": null,
        "audio_path": null,
        "audio_script_gold": null,
        "speech_effects": null,
        "source_effects": null,
        "channel_effects": null,
        "turn_taking_action": null,
        "utterance_ids": null,
        "chunk_id": null,
        "is_final_chunk": true,
        "source": null,
        "contains_speech": true,
    },
    {
        "role": "user",
        "content": "Understood on **H9ZU1C**—since you can’t remove Ethan and cancellation isn’t allowed under the criteria you listed, please **don’t cancel** and **no transfer needed**. I’ll move on.\n\nFor the new trip:\n- **Passenger count:** just **1** (me).\n- **West Coast airports:** any of these are fine: **LAX, SFO, SEA, SAN, LAS**.\n- Please book the **cheapest direct round trip** for **May 20 → May 25** from **JFK or EWR**, **basic economy** if that’s cheapest.\n- Apply payment using the **$39 gift card first**, then the **$203 gift card**.\n- **No insurance.**\n- For bags: I want to use **all free baggage allowance** (but **don’t add any paid bags**).\n\nAlso, please use my **DOB from my profile** (don’t need me to type it).",
        "tool_calls": null,
        "is_audio": false,
        "turn_idx": 20,
        "timestamp": "2026-05-24T16:35:31.671100",
        "cost": 0.00539,
        "usage": {"completion_tokens": 202, "prompt_tokens": 1464},
        "raw_data": {
            "id": "chatcmpl-Dj9pg6zAKt09v7me0RBtK0YVYzrYx",
            "created": 1779654928,
            "model": "gpt-5.2-2025-12-11",
            "object": "chat.completion",
            "system_fingerprint": null,
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Understood on **H9ZU1C**—since you can’t remove Ethan and cancellation isn’t allowed under the criteria you listed, please **don’t cancel** and **no transfer needed**. I’ll move on.\n\nFor the new trip:\n- **Passenger count:** just **1** (me).\n- **West Coast airports:** any of these are fine: **LAX, SFO, SEA, SAN, LAS**.\n- Please book the **cheapest direct round trip** for **May 20 → May 25** from **JFK or EWR**, **basic economy** if that’s cheapest.\n- Apply payment using the **$39 gift card first**, then the **$203 gift card**.\n- **No insurance.**\n- For bags: I want to use **all free baggage allowance** (but **don’t add any paid bags**).\n\nAlso, please use my **DOB from my profile** (don’t need me to type it).",
                        "role": "assistant",
                        "tool_calls": null,
                        "function_call": null,
                        "provider_specific_fields": {"refusal": null},
                        "annotations": [],
                    },
                    "provider_specific_fields": {},
                }
            ],
            "usage": {
                "completion_tokens": 202,
                "prompt_tokens": 1464,
                "total_tokens": 1666,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            },
            "service_tier": "default",
        },
        "generation_time_seconds": null,
        "audio_format": null,
        "audio_path": null,
        "audio_script_gold": null,
        "speech_effects": null,
        "source_effects": null,
        "channel_effects": null,
        "turn_taking_action": null,
        "utterance_ids": null,
        "chunk_id": null,
        "is_final_chunk": true,
        "source": null,
        "contains_speech": true,
    },
    {
        "role": "assistant",
        "content": null,
        "tool_calls": [
            {
                "id": "call_d7zYZAFGISouoidvpNor6gfE",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "JFK",
                    "destination": "LAX",
                    "date": "2024-05-25",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_inqZ4R69BaYfX83NMOsMI4Z6",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "JFK",
                    "destination": "SFO",
                    "date": "2024-05-25",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_SNGz4QeMtsrd8lxOGxY1pDZV",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "JFK",
                    "destination": "SEA",
                    "date": "2024-05-25",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_aOvBNamEYl01W2CJgEJs0wZi",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "JFK",
                    "destination": "SAN",
                    "date": "2024-05-25",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_1qNYfshQWDl0aiwqtJSeNkUH",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "JFK",
                    "destination": "LAS",
                    "date": "2024-05-25",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_1tEyoK7HjO14hojkZp3zQFcB",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "EWR",
                    "destination": "LAX",
                    "date": "2024-05-25",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_DLFvmDKrNrA3D4kBSurBz8FD",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "EWR",
                    "destination": "SFO",
                    "date": "2024-05-25",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_hQOQ9XbE7oh7DSLfVlGdk9iY",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "EWR",
                    "destination": "SEA",
                    "date": "2024-05-25",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_Ikeabe2wS0oecMdYQ1JRfxYY",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "EWR",
                    "destination": "SAN",
                    "date": "2024-05-25",
                },
                "requestor": "assistant",
            },
            {
                "id": "call_qKN3aHd9B7pOvnKwMqbADdGm",
                "name": "search_direct_flight",
                "arguments": {
                    "origin": "EWR",
                    "destination": "LAS",
                    "date": "2024-05-25",
                },
                "requestor": "assistant",
            },
        ],
        "is_audio": false,
        "turn_idx": 21,
        "timestamp": "2026-05-24T16:35:37.919903",
        "cost": 0.0077294,
        "usage": {"completion_tokens": 323, "prompt_tokens": 5980},
        "raw_data": {
            "id": "chatcmpl-Dj9pmc5s7ArhUzt560emKAl1oPfAG",
            "created": 1779654934,
            "model": "gpt-5.2-2025-12-11",
            "object": "chat.completion",
            "system_fingerprint": null,
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "content": null,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": '{"origin": "JFK", "destination": "LAX", "date": "2024-05-25"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_d7zYZAFGISouoidvpNor6gfE",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "JFK", "destination": "SFO", "date": "2024-05-25"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_inqZ4R69BaYfX83NMOsMI4Z6",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "JFK", "destination": "SEA", "date": "2024-05-25"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_SNGz4QeMtsrd8lxOGxY1pDZV",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "JFK", "destination": "SAN", "date": "2024-05-25"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_aOvBNamEYl01W2CJgEJs0wZi",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "JFK", "destination": "LAS", "date": "2024-05-25"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_1qNYfshQWDl0aiwqtJSeNkUH",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "EWR", "destination": "LAX", "date": "2024-05-25"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_1tEyoK7HjO14hojkZp3zQFcB",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "EWR", "destination": "SFO", "date": "2024-05-25"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_DLFvmDKrNrA3D4kBSurBz8FD",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "EWR", "destination": "SEA", "date": "2024-05-25"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_hQOQ9XbE7oh7DSLfVlGdk9iY",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "EWR", "destination": "SAN", "date": "2024-05-25"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_Ikeabe2wS0oecMdYQ1JRfxYY",
                                "type": "function",
                            },
                            {
                                "function": {
                                    "arguments": '{"origin": "EWR", "destination": "LAS", "date": "2024-05-25"}',
                                    "name": "search_direct_flight",
                                },
                                "id": "call_qKN3aHd9B7pOvnKwMqbADdGm",
                                "type": "function",
                            },
                        ],
                        "function_call": null,
                        "provider_specific_fields": {"refusal": null},
                        "annotations": [],
                    },
                    "provider_specific_fields": {},
                }
            ],
            "usage": {
                "completion_tokens": 323,
                "prompt_tokens": 5980,
                "total_tokens": 6303,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 4608},
            },
            "service_tier": "default",
            "guidance": [],
        },
        "generation_time_seconds": 3.4402082519955,
        "audio_format": null,
        "audio_path": null,
        "audio_script_gold": null,
        "speech_effects": null,
        "source_effects": null,
        "channel_effects": null,
        "turn_taking_action": null,
        "utterance_ids": null,
        "chunk_id": null,
        "is_final_chunk": true,
        "source": null,
        "contains_speech": true,
    },
    {
        "id": "call_d7zYZAFGISouoidvpNor6gfE",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 22,
        "timestamp": "2026-05-24T16:35:37.920158",
    },
    {
        "id": "call_inqZ4R69BaYfX83NMOsMI4Z6",
        "role": "tool",
        "content": '[{"flight_number": "HAT023", "origin": "JFK", "destination": "SFO", "status": "available", "scheduled_departure_time_est": "14:00:00", "scheduled_arrival_time_est": "20:00:00", "date": null, "available_seats": {"basic_economy": 9, "economy": 18, "business": 0}, "prices": {"basic_economy": 99, "economy": 159, "business": 202}}]',
        "requestor": "assistant",
        "error": false,
        "turn_idx": 23,
        "timestamp": "2026-05-24T16:35:37.920226",
    },
    {
        "id": "call_SNGz4QeMtsrd8lxOGxY1pDZV",
        "role": "tool",
        "content": '[{"flight_number": "HAT069", "origin": "JFK", "destination": "SEA", "status": "available", "scheduled_departure_time_est": "06:00:00", "scheduled_arrival_time_est": "12:00:00", "date": null, "available_seats": {"basic_economy": 5, "economy": 12, "business": 13}, "prices": {"basic_economy": 66, "economy": 120, "business": 444}}, {"flight_number": "HAT083", "origin": "JFK", "destination": "SEA", "status": "available", "scheduled_departure_time_est": "01:00:00", "scheduled_arrival_time_est": "07:00:00", "date": null, "available_seats": {"basic_economy": 17, "economy": 14, "business": 19}, "prices": {"basic_economy": 93, "economy": 116, "business": 223}}]',
        "requestor": "assistant",
        "error": false,
        "turn_idx": 24,
        "timestamp": "2026-05-24T16:35:37.920296",
    },
    {
        "id": "call_aOvBNamEYl01W2CJgEJs0wZi",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 25,
        "timestamp": "2026-05-24T16:35:37.920322",
    },
    {
        "id": "call_1qNYfshQWDl0aiwqtJSeNkUH",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 26,
        "timestamp": "2026-05-24T16:35:37.920344",
    },
    {
        "id": "call_1tEyoK7HjO14hojkZp3zQFcB",
        "role": "tool",
        "content": '[{"flight_number": "HAT041", "origin": "EWR", "destination": "LAX", "status": "available", "scheduled_departure_time_est": "07:00:00", "scheduled_arrival_time_est": "12:30:00", "date": null, "available_seats": {"basic_economy": 0, "economy": 17, "business": 1}, "prices": {"basic_economy": 89, "economy": 110, "business": 397}}]',
        "requestor": "assistant",
        "error": false,
        "turn_idx": 27,
        "timestamp": "2026-05-24T16:35:37.920380",
    },
    {
        "id": "call_DLFvmDKrNrA3D4kBSurBz8FD",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 28,
        "timestamp": "2026-05-24T16:35:37.920400",
    },
    {
        "id": "call_hQOQ9XbE7oh7DSLfVlGdk9iY",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 29,
        "timestamp": "2026-05-24T16:35:37.920420",
    },
    {
        "id": "call_Ikeabe2wS0oecMdYQ1JRfxYY",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 30,
        "timestamp": "2026-05-24T16:35:37.920440",
    },
    {
        "id": "call_qKN3aHd9B7pOvnKwMqbADdGm",
        "role": "tool",
        "content": "[]",
        "requestor": "assistant",
        "error": false,
        "turn_idx": 31,
        "timestamp": "2026-05-24T16:35:37.920458",
    },
    {
        "role": "assistant",
        "content": "Cheapest direct round-trip I can build (May 20 → May 25), from **JFK/EWR** to West Coast, **Basic Economy**, **1 passenger (Mia Kim, DOB 1965-06-09)**, **no insurance**:\n\n## Cheapest option found: **JFK ↔ SEA**\n**Outbound (May 20):**\n- **HAT069** — **JFK → SEA**, dep **06:00**, arr **12:00**\n- Price (Basic Economy): **$51**\n\n**Return (May 25):**\n- **HAT069** — **JFK → SEA**, dep **06:00**, arr **12:00**\n- Price (Basic Economy): **$66**\n\n**Estimated flight total:** **$117** (51 + 66)\n\n## Bags (free only, no paid bags)\nYou’re **Silver**. In **Basic Economy**, that includes **1 free checked bag per passenger**.\n- You asked to use *all* free allowance and add **no paid bags**, so I will set:\n  - **Total checked bags: 1**\n  - **Non-free (paid) bags: 0**\n\n## Payment (smaller gift card first, then larger)\nYou have:\n- **gift_card_7359776**: $39\n- **gift_card_7773485**: $203\n\nI will apply:\n- **$39** from **gift_card_7359776**\n- **$78** from **gift_card_7773485**\n(Total paid: $117)\n\n---\n\n### Please confirm (reply “yes”) to book with these exact details:\n- User: **mia_kim_4397**\n- Trip type: **round trip**\n- Cabin: **basic_economy**\n- Flights:\n  - **HAT069** on **2024-05-20** (JFK → SEA)\n  - **HAT069** on **2024-05-25** (JFK → SEA)\n- Passenger:\n  - **Mia Kim**, **DOB 1965-06-09**\n- Bags: **1 total**, **0 paid**\n- Insurance: **no**\n- Payment:\n  - **gift_card_7359776: $39**\n  - **gift_card_7773485: $78**",
        "tool_calls": null,
        "is_audio": false,
        "turn_idx": 32,
        "timestamp": "2026-05-24T16:35:44.862776",
        "cost": 0.009788100000000001,
        "usage": {"completion_tokens": 485, "prompt_tokens": 6782},
        "raw_data": {
            "id": "chatcmpl-Dj9pqmn2z8k6YfFojvwWVjmWkadY0",
            "created": 1779654938,
            "model": "gpt-5.2-2025-12-11",
            "object": "chat.completion",
            "system_fingerprint": null,
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Cheapest direct round-trip I can build (May 20 → May 25), from **JFK/EWR** to West Coast, **Basic Economy**, **1 passenger (Mia Kim, DOB 1965-06-09)**, **no insurance**:\n\n## Cheapest option found: **JFK ↔ SEA**\n**Outbound (May 20):**\n- **HAT069** — **JFK → SEA**, dep **06:00**, arr **12:00**\n- Price (Basic Economy): **$51**\n\n**Return (May 25):**\n- **HAT069** — **JFK → SEA**, dep **06:00**, arr **12:00**\n- Price (Basic Economy): **$66**\n\n**Estimated flight total:** **$117** (51 + 66)\n\n## Bags (free only, no paid bags)\nYou’re **Silver**. In **Basic Economy**, that includes **1 free checked bag per passenger**.\n- You asked to use *all* free allowance and add **no paid bags**, so I will set:\n  - **Total checked bags: 1**\n  - **Non-free (paid) bags: 0**\n\n## Payment (smaller gift card first, then larger)\nYou have:\n- **gift_card_7359776**: $39\n- **gift_card_7773485**: $203\n\nI will apply:\n- **$39** from **gift_card_7359776**\n- **$78** from **gift_card_7773485**\n(Total paid: $117)\n\n---\n\n### Please confirm (reply “yes”) to book with these exact details:\n- User: **mia_kim_4397**\n- Trip type: **round trip**\n- Cabin: **basic_economy**\n- Flights:\n  - **HAT069** on **2024-05-20** (JFK → SEA)\n  - **HAT069** on **2024-05-25** (JFK → SEA)\n- Passenger:\n  - **Mia Kim**, **DOB 1965-06-09**\n- Bags: **1 total**, **0 paid**\n- Insurance: **no**\n- Payment:\n  - **gift_card_7359776: $39**\n  - **gift_card_7773485: $78**",
                        "role": "assistant",
                        "tool_calls": null,
                        "function_call": null,
                        "provider_specific_fields": {"refusal": null},
                        "annotations": [],
                    },
                    "provider_specific_fields": {},
                }
            ],
            "usage": {
                "completion_tokens": 485,
                "prompt_tokens": 6782,
                "total_tokens": 7267,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 5632},
            },
            "service_tier": "default",
            "guidance": [],
        },
        "generation_time_seconds": 6.550092620018404,
        "audio_format": null,
        "audio_path": null,
        "audio_script_gold": null,
        "speech_effects": null,
        "source_effects": null,
        "channel_effects": null,
        "turn_taking_action": null,
        "utterance_ids": null,
        "chunk_id": null,
        "is_final_chunk": true,
        "source": null,
        "contains_speech": true,
    },
]

tau2_messages = to_tau2_messages([system] + messages)

print(trustworthiness_from_messages(tau2_messages[:-1], tau2_messages[-1], tools, TLM())["trustworthiness_score"])  # type: ignore
exit()

# response = generate(
#     model="gpt-5.2",
#     messages=tau2_messages,
#     tools=tools,
# )

# print(loads(response.content)["message"] if response.content else response.tool_calls)  # type: ignore

# if response.content:
#     print(trustworthiness_from_messages(tau2_messages, response, tools, TLM())["trustworthiness_score"])  # type: ignore
