import { AgentSquad } from "./orchestrator";
import { AnthropicAgent } from "./agents/anthropicAgent";
import { AgentResponse } from "./agents/agent";

const orchestrator = new AgentSquad({
  config: {
    LOG_AGENT_CHAT: true,
    LOG_CLASSIFIER_CHAT: true,
    LOG_CLASSIFIER_RAW_OUTPUT: false,
    LOG_CLASSIFIER_OUTPUT: true,
    LOG_EXECUTION_TIMES: true,
  },
});

orchestrator.addAgent(
  new AnthropicAgent({
    name: "Tech Agent",
    description:
      "Specializes in technology areas including software development, hardware, AI, cybersecurity, blockchain, cloud computing, emerging tech innovations, and pricing/costs related to technology products and services.",
    inferenceConfig: {
      maxTokens: 2500,
      temperature: 1,
      topP: 0.96
    },
    modelId: "claude-3-7-sonnet-20250219",
    thinking: {
      type: "enabled",
      budget_tokens: 1024,
    },
    streaming: true,
    apiKey:
      "REPLACE",
  })
);

const userId = "quickstart-user";
// set guid as sessionId
const sessionId = `${Date.now()}`;
const query = "What are the latest trends in AI?";
console.log(`\nUser Query: ${query}`);

async function main() {
  try {
    const response = await orchestrator.routeRequest(query, userId, sessionId);

    // Handle streaming response
    if (response.streaming == true) {
      console.log("\n** STREAMING RESPONSE ** \n");
      // Stream the content
      for await (const chunk of response.output) {
        if (typeof chunk === "string") {
          process.stdout.write(chunk);
        } else {
          console.error("Received unexpected chunk type:", typeof chunk);
        }
      }
      console.log("\n");
    } else {
      // Handle non-streaming response
      const agentResponse = response as AgentResponse;
      console.log("\n** RESPONSE ** \n");
      console.log(`> Agent ID: ${agentResponse.metadata.agentId}`);
      console.log(`> Agent Name: ${agentResponse.metadata.agentName}`);
      console.log(`> User Input: ${agentResponse.metadata.userInput}`);
      console.log(`> User ID: ${agentResponse.metadata.userId}`);
      console.log(`> Session ID: ${agentResponse.metadata.sessionId}`);
      console.log(
        `> Additional Parameters:`,
        agentResponse.metadata.additionalParams
      );
      console.log(`\nThinking:\n${agentResponse.thinking}`);
      console.log(`\nResponse:\n${agentResponse.output}`);
    }
  } catch (error) {
    console.error("An error occurred:", error);
  }
}

main();
