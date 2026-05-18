from typing import List, Optional

from tlm import TLM
from tlm.config.base import ConfigInput

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import (
    LLMAgent,
    LLMAgentState,
)
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate
from tau2.utils.trustworthiness import (
    determine_rewrite,
    get_fix_messages,
    trustworthiness_from_messages,
)


class TLMAgent(LLMAgent):
    """
    An LLM agent that can be used to solve a task.
    """

    tlm: TLM

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the TLMAgent.
        """
        super().__init__(
            tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args
        )
        self.tlm = TLM(
            config_input=ConfigInput(
                reasoning_effort="medium",  # type: ignore
            )
        )

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> AssistantMessage:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.messages

        assistant_message: AssistantMessage = generate(  # type: ignore
            model=self.llm,
            tools=self.tools,
            messages=messages,  # type: ignore
            call_name="agent_response",
            **self.llm_args,
        )

        # if (
        #     not assistant_message.has_text_content()
        #     and not assistant_message.is_tool_call()
        # ):
        #     assistant_message.content = "Could you please clarify?"

        trustworthiness = trustworthiness_from_messages(
            messages, assistant_message, self.tools, self.tlm
        )

        if trustworthiness["confidence_score"] < 0.75:  # type: ignore
            fix_messages = get_fix_messages(assistant_message, trustworthiness)

            new_assistant_message: AssistantMessage = generate(  # type: ignore
                model=self.llm,
                tools=self.tools,
                messages=messages + [assistant_message] + fix_messages,  # type: ignore
                call_name="agent_response",
                **self.llm_args,
            )

            new_trustworthiness = trustworthiness_from_messages(
                messages, new_assistant_message, self.tools, self.tlm
            )

            total_cost = new_assistant_message.cost + assistant_message.cost  # type: ignore
            assistant_message.cost = total_cost
            new_assistant_message.cost = total_cost

            assistant_message = determine_rewrite(
                assistant_message,
                new_assistant_message,
                rewrite=(
                    new_trustworthiness["confidence_score"]
                    > trustworthiness["confidence_score"] + 0.1
                ),
            )

        return assistant_message


def create_tlm_agent(tools, domain_policy, **kwargs):
    """Factory function for TLMAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
    """
    return TLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),  # type: ignore
        llm_args=kwargs.get("llm_args"),
    )
