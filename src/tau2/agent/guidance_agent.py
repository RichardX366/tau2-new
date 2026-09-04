from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import (
    LLMAgent,
    LLMAgentState,
)
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
)
from tau2.utils.guidance import (
    get_cancel_tool_messages,
    get_post_guidance_message,
    get_pre_guidance_message,
)
from tau2.utils.llm_utils import generate


class GuidanceAgent(LLMAgent):
    """
    An LLM agent that can be used to solve a task.
    """

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

        guidance, guidance_message = get_pre_guidance_message(messages)

        assistant_message: AssistantMessage = generate(  # type: ignore
            model=self.llm,
            tools=self.tools,
            messages=messages + guidance_message,  # type: ignore
            call_name="agent_response",
            **self.llm_args,
        )

        assistant_message.raw_data["guidance"] = guidance  # type: ignore

        post_guidance, post_guidance_message = get_post_guidance_message(
            messages + [assistant_message]
        )

        if post_guidance:
            assistant_message = generate(  # type: ignore
                model=self.llm,
                tools=self.tools,
                messages=messages + [assistant_message] + get_cancel_tool_messages(assistant_message) + post_guidance_message,  # type: ignore
                call_name="agent_response",
                **self.llm_args,
            )
            assistant_message.raw_data["guidance"] = guidance + post_guidance  # type: ignore

        return assistant_message


def create_guidance_agent(tools, domain_policy, **kwargs):
    """Factory function for TLMAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
    """
    return GuidanceAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),  # type: ignore
        llm_args=kwargs.get("llm_args"),
    )
