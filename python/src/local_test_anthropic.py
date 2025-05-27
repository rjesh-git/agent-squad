import uuid
import asyncio
from typing import Any
import nest_asyncio


from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
from agent_squad.agents import (
    AnthropicAgent,
    AnthropicAgentOptions,
    AgentResponse,
    AgentCallbacks,
)
from agent_squad.types import ConversationMessage


class LLMAgentCallbacks(AgentCallbacks):
    think_state: str = "not-started"

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        # handle response streaming here
        if "thinking" in kwargs:
            if self.think_state == "not-started":
                print("Thinking:\n---------\n", end="", flush=True)
                self.think_state = "started"
            print(token, end="", flush=True)
        else:
            if self.think_state != "finished":
                print("\nResponse:\n---------\n", end="", flush=True)
                self.think_state = "finished"
            print(token, end="", flush=True)

    async def on_llm_end(self, name: str, output: Any, **kwargs) -> None:
        print("\n----.:: End ::.----\n")


# Initialize the orchestrator with some options
orchestrator = AgentSquad(
    options=AgentSquadConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_CLASSIFIER_RAW_OUTPUT=True,
        LOG_CLASSIFIER_OUTPUT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MAX_MESSAGE_PAIRS_PER_AGENT=10,
    )
)

# Add some agents
tech_agent = AnthropicAgent(
    AnthropicAgentOptions(
        name="Tech Agent",
        api_key="REDACTED",
        streaming=True,
        description="Specializes in technology areas including software development, hardware, AI, \
        cybersecurity, blockchain, cloud computing, emerging tech innovations, and pricing/costs \
        related to technology products and services.",
        model_id="claude-3-7-sonnet-20250219",
        callbacks=LLMAgentCallbacks(),
        inference_config={"maxTokens": 2500, "temperature": 1, "topP": 0.95},
        thinking={"type": "enabled", "budget_tokens": 2000},
    )
)
orchestrator.add_agent(tech_agent)


async def handle_request(_orchestrator: AgentSquad, _user_input: str, _user_id: str, _session_id: str):
    try:
        response: AgentResponse = await _orchestrator.route_request(_user_input, _user_id, _session_id)

        # Print metadata
        print("\nMetadata:")
        print(f"Selected Agent: {response.metadata.agent_name}")
        if isinstance(response, AgentResponse) and response.streaming is False:
            # Handle regular response
            if isinstance(response.output, str):
                print(f"string:\n\n{response.output}\n\n")
            elif isinstance(response.output, ConversationMessage):
                # Handle different content types in the response
                for content_block in response.output.content:
                    if hasattr(content_block, "thinking"):
                        print("\nThinking:\n---------\n")
                        print(content_block.thinking)
                    elif hasattr(content_block, "text"):
                        print("\nResponse:\n---------\n")
                        print(content_block.text)
        return response
    except Exception as e:
        print(f"Error in handle_request: {str(e)}")
        return None


if __name__ == "__main__":
    USER_ID = "user123"
    SESSION_ID = str(uuid.uuid4())

    nest_asyncio.apply()

    user_input = "what is blockchain"

    # Run the async function
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))
