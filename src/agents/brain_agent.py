import os
import sys
import json
from src.exception import CustomException
from src.logger import logging

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

class BrainAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.environ.get("GROQ_API_KEY")
        )

    def detect_intent(self, user_message):
        """Analyze user's text to determine if they want crop classification or disease detection."""
        try:
            messages = [
                SystemMessage(content=(
                    "You are an intent classifier for an agriculture AI system. "
                    "Based on the user's message, determine if they want:\n"
                    "1. 'crop' - to identify what type of crop is in the image (e.g., rice, wheat, maize, jute, sugarcane)\n"
                    "2. 'disease' - to detect disease in a crop/leaf image\n"
                    "3. 'general' - general agriculture question (no image classification needed)\n\n"
                    "Respond with ONLY a JSON object: {\"intent\": \"crop\"} or {\"intent\": \"disease\"} or {\"intent\": \"general\"}\n"
                    "No other text."
                )),
                HumanMessage(content=user_message)
            ]

            response = self.llm.invoke(messages)
            result = json.loads(response.content.strip())
            intent = result.get("intent", "general")
            logging.info(f"Brain agent detected intent: {intent} for message: {user_message}")
            return intent

        except Exception as e:
            logging.warning(f"Intent detection failed: {e}. Defaulting to 'crop'")
            return "crop"

    def format_response(self, user_message, intent, classification_result):
        """Use Grok to generate a natural language response based on classifier output."""
        try:
            if intent == "crop":
                context = (
                    f"The crop classifier identified the image as: {classification_result['crop']} "
                    f"with {classification_result['confidence']}% confidence."
                )
            elif intent == "disease":
                context = (
                    f"The disease detector identified: {classification_result['disease']} "
                    f"with {classification_result['confidence']}% confidence."
                )
            else:
                context = ""

            messages = [
                SystemMessage(content=(
                    "You are CropAI, a friendly agriculture AI assistant. "
                    "You help farmers identify crops and detect plant diseases. "
                    "Give a brief, helpful response based on the classification result provided. "
                    "Keep it concise (2-3 sentences max). "
                    "If it's a disease, briefly mention what the farmer should look out for."
                )),
                HumanMessage(content=(
                    f"User asked: {user_message}\n\n"
                    f"Classification result: {context}\n\n"
                    f"Provide a helpful response."
                ))
            ]

            response = self.llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logging.warning(f"Response formatting failed: {e}")
            if intent == "crop":
                return f"I identified this as **{classification_result['crop']}** with {classification_result['confidence']}% confidence."
            elif intent == "disease":
                return f"I detected **{classification_result['disease']}** with {classification_result['confidence']}% confidence."
            return "I couldn't process your request. Please try again."

    def handle_general_query(self, user_message):
        """Handle general agriculture questions without image classification."""
        try:
            messages = [
                SystemMessage(content=(
                    "You are CropAI, a friendly agriculture AI assistant. "
                    "Answer agriculture-related questions concisely (2-3 sentences). "
                    "If the question requires an image but none was provided, ask the user to upload one."
                )),
                HumanMessage(content=user_message)
            ]

            response = self.llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            raise CustomException(e, sys)
