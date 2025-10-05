import json
import random
import requests
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
import os

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Environment variables
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "default-model")


class CDSCChatbot:
    def __init__(self, intents_file='intents.json'):
        """Initialize CDSC Enhanced Chatbot with semantic search."""
        logger.info("Initializing CDSC Enhanced Chatbot...")
        self.intents = self.load_intents(intents_file)
        self.intent_text_cache = self._flatten_patterns()
        self.embeddings_cache = {}
        logger.info(f"Loaded {len(self.intents)} intents successfully.")

    # -------------------- INTENT LOADING --------------------
    def load_intents(self, filename):
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                return data.get('intents', [])
        except Exception as e:
            logger.error(f"Error loading intents: {e}")
            return []

    def _flatten_patterns(self):
        """Flatten all intent patterns for semantic search."""
        intent_map = {}
        for intent in self.intents:
            if intent.get("patterns"):
                intent_map[intent["tag"]] = intent["patterns"]
        return intent_map

    # -------------------- EMBEDDING / SEMANTIC MATCH --------------------
    def get_embedding(self, text):
        """Fetch embedding vector from API (OpenAI/Gemini compatible)."""
        try:
            payload = {"model": MODEL_NAME, "input": text}
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = requests.post(f"{API_URL}/embeddings", headers=headers, json=payload, timeout=15)
            data = response.json()
            return np.array(data["data"][0]["embedding"])
        except Exception as e:
            logger.warning(f"Embedding fetch failed: {e}")
            return np.zeros(768)  # fallback zero vector

    def find_best_intent(self, user_message):
        """Find the closest intent tag based on embedding similarity."""
        user_emb = self.get_embedding(user_message)
        best_tag, best_score = None, -1

        for tag, patterns in self.intent_text_cache.items():
            for pattern in patterns:
                if pattern not in self.embeddings_cache:
                    self.embeddings_cache[pattern] = self.get_embedding(pattern)
                sim = cosine_similarity([user_emb], [self.embeddings_cache[pattern]])[0][0]
                if sim > best_score:
                    best_score, best_tag = sim, tag

        if best_tag:
            logger.info(f"Best semantic match found: {best_tag} (score: {best_score:.3f})")
            return best_tag, best_score
        return "fallback", 0.0

    # -------------------- RESPONSE GENERATION --------------------
    def generate_detailed_response(self, user_message, intent_tag):
        """Produce a detailed, human-like reply using API."""
        intent = next((i for i in self.intents if i.get("tag") == intent_tag), None)
        style_examples = ", ".join(intent.get("responses", [])) if intent else ""

        system_prompt = (
            "You are CDSC Club’s highly knowledgeable and friendly chatbot. "
            "Always reply helpfully, in natural conversational language, "
            "adding context, examples, and clarity. "
            "If user input is slightly ambiguous, infer the most likely intent and respond correctly. "
            "Avoid robotic, vague, or repetitive answers. "
            "Provide detailed, accurate, and user-friendly guidance."
        )

        prompt = (
            f"User message: {user_message}\n"
            f"Identified intent: {intent_tag}\n"
            f"Example responses: {style_examples}\n\n"
            "Craft a detailed, helpful, friendly, and context-aware response."
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }

        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            data = response.json()
            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not reply:
                reply = random.choice(intent.get("responses", [])) if intent else "I'm here to help!"
            return reply
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "Sorry, something went wrong while generating a detailed reply."

    # -------------------- MAIN SEMANTIC PIPELINE --------------------
    def api_semantic_match(self, user_message):
        """Full pipeline: find intent, generate detailed response, fallback if needed."""
        try:
            tag, score = self.find_best_intent(user_message)
            if score < 0.6:  # confidence threshold
                return self.get_fallback_intent()

            detailed_reply = self.generate_detailed_response(user_message, tag)
            return {"tag": tag, "confidence": float(score), "response": detailed_reply}

        except Exception as e:
            logger.error(f"Semantic pipeline failed: {e}")
            return self.get_fallback_intent()

    # -------------------- FALLBACK --------------------
    def get_fallback_intent(self):
        """Default fallback response if no intent matches."""
        fallback = next((i for i in self.intents if i.get("tag") == "fallback"), None)
        return {
            "tag": "fallback",
            "confidence": 0.0,
            "response": random.choice(fallback.get("responses", ["I'm not sure I understand yet."])) if fallback else "I'm not sure I understand yet."
        }

    # -------------------- TRAINING EXAMPLES --------------------
    def add_training_example(self, user_message, correct_intent):
        """Record new user message as training pattern."""
        for intent in self.intents:
            if intent.get("tag") == correct_intent:
                intent.setdefault("patterns", []).append(user_message)
                logger.info(f"Added training example: '{user_message}' -> {correct_intent}")
                break
