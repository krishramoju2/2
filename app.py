class CDSCChatbot:
    def __init__(self, intents_file='intents.json'):
        logger.info("Initializing Enhanced CDSC Chatbot...")
        self.intents = self.load_intents(intents_file)
        self.intent_text_cache = self._flatten_patterns()
        self.embeddings_cache = {}
        logger.info(f"Loaded {len(self.intents)} intents with improved semantic logic.")

    def load_intents(self, filename):
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                return data['intents']
        except Exception as e:
            logger.error(f"Error loading intents: {e}")
            return []

    def _flatten_patterns(self):
        intent_map = {}
        for intent in self.intents:
            if intent["patterns"]:
                intent_map[intent["tag"]] = intent["patterns"]
        return intent_map

    # -------------------- NEW SEMANTIC MATCHING --------------------

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
        """Use embedding cosine similarity to pick the closest intent."""
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
            logger.info(f"Best semantic match: {best_tag} ({best_score:.3f})")
            return best_tag, best_score
        return "fallback", 0.0

    # -------------------- RESPONSE GENERATION --------------------

    def generate_detailed_response(self, user_message, intent_tag):
        """Use external API to produce a detailed, human-like reply."""
        intent = next((i for i in self.intents if i["tag"] == intent_tag), None)
        style_examples = ", ".join(intent["responses"]) if intent else ""

        system_prompt = (
            "You are CDSC Club’s friendly and knowledgeable assistant. "
            "You always reply helpfully, in a natural conversational tone, "
            "adding a bit of context or elaboration when possible. "
            "Avoid being repetitive or robotic."
        )

        prompt = (
            f"The user's message: {user_message}\n"
            f"The identified intent: {intent_tag}\n"
            f"Example style responses: {style_examples}\n\n"
            "Now write a detailed and friendly response that fits the context."
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            data = response.json()
            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not reply:
                reply = random.choice(intent["responses"]) if intent else "I'm here to help!"
            return reply
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "Sorry, something went wrong while generating a detailed reply."

    # -------------------- MAIN PIPELINE --------------------

    def api_semantic_match(self, user_message):
        try:
            tag, score = self.find_best_intent(user_message)
            if score < 0.6:  # low confidence threshold
                return self.get_fallback_intent()

            detailed_reply = self.generate_detailed_response(user_message, tag)
            return {"tag": tag, "confidence": float(score), "response": detailed_reply}

        except Exception as e:
            logger.error(f"Semantic pipeline failed: {e}")
            return self.get_fallback_intent()
