import os
from src.logger import logging

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from src.agents.tools import (
    analyze_image,
    answer_crop_question,
    set_image_path,
)

SYSTEM_PROMPT = """You are CropAI, an agriculture assistant for crop identification, disease detection (Rice & Sugarcane only), and farming questions.

TOOLS AVAILABLE:
1. analyze_image(query="image") — call whenever an image is uploaded. Always pass the string "image" as query.
2. answer_crop_question(question="...") — call for Rice or Sugarcane farming questions. Pass the user's exact question as the argument.

WHEN NOT TO USE ANY TOOL:
- Greetings, thanks, farewells, casual chat → reply naturally and briefly. Do NOT mention farming unless user asks.

WHEN TO USE TOOLS:
- Image uploaded → call analyze_image(query="image")
- User asks a farming question about Rice or Sugarcane (cultivation, diseases, irrigation, pests, fertilizers, harvesting) → call answer_crop_question(question="<user question>")
- User asks about Jute, Maize, or Wheat → answer from your own knowledge, do NOT call any tool
- User says "yes" to see disease list but no image is uploaded → show list only, do NOT call any tool

INTERPRETING analyze_image RESULTS:
The tool returns two lines:
  Crop Classifier  → [Crop] ([X]%)
  Disease Classifier → [Disease] ([Y]%) [This disease belongs to: Rice/Sugarcane]

Disease-to-crop facts (never override):
  Rice diseases: Bacterial Blight, Blast, Brown Spot, Tungro
  Sugarcane diseases: Healthy, Mosaic, Red Rot, Rust, Yellow

Logic:
- Crop=Rice or Sugarcane → report crop + disease, note if they match
- Crop=Maize/Jute/Wheat BUT disease belongs to Rice or Sugarcane → the crop classifier misread a close-up leaf; trust the disease result and correct the crop name
- Crop=Maize/Jute/Wheat AND disease is also unclear → report crop only; say disease detection is for Rice and Sugarcane only

DISEASE LISTS (show only when user says yes and no image is uploaded):
Rice: Bacterial Blight, Blast, Brown Spot, Tungro
Sugarcane: Mosaic, Red Rot, Rust, Yellow, Healthy

Be concise and farmer-friendly in every response."""


class BrainAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            max_tokens=400,
        )

        tools = [analyze_image, answer_crop_question]

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
        )

    def _fallback_response(self, user_message: str) -> str:
        """Direct LLM call with no tools — used when the agent executor fails."""
        logging.warning("Using fallback direct LLM (no tools)")
        try:
            response = self.llm.invoke([
                SystemMessage(content=(
                    "You are CropAI, a helpful agriculture assistant specialising in "
                    "rice and sugarcane farming. Answer concisely and accurately."
                )),
                HumanMessage(content=user_message),
            ])
            return response.content
        except Exception as fallback_err:
            logging.error(f"Fallback LLM also failed: {fallback_err}")
            return "Sorry, I ran into a temporary issue. Please try again."

    def run(self, user_message, chat_history, image_path=None):
        """Run the agent with conversation history and optional image path."""
        set_image_path(image_path)

        if image_path:
            augmented_message = (
                f"[Image uploaded — call analyze_image(query='image')]\n"
                f"User: {user_message}"
            )
        else:
            augmented_message = user_message

        trimmed_history = chat_history[-6:] if len(chat_history) > 6 else chat_history

        logging.info(f"Agent input: {augmented_message}")

        try:
            result = self.executor.invoke({
                "input": augmented_message,
                "chat_history": trimmed_history,
            })
            output = result["output"]

            # Guard: AgentExecutor returns a plain string like "Agent stopped due to
            # max iterations." when it gives up. Never surface that to the user.
            if isinstance(output, str) and (
                output.startswith("Agent stopped")
                or "max iterations" in output.lower()
            ):
                logging.warning(f"Agent hit iteration limit: {output!r}")
                return self._fallback_response(user_message)

            logging.info(f"Agent output: {output}")
            return output

        except Exception as e:
            logging.warning(f"Agent executor failed ({type(e).__name__}: {e})")
            return self._fallback_response(user_message)
