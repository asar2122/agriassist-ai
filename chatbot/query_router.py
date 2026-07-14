class QueryRouter:

    RAG_INTENTS = {
        "crop",
        "disease",
        "fertilizer",
        "pest",
        "general",
    }

    def should_use_rag(self, intent):
        return intent in self.RAG_INTENTS

    def route(self, intent):

        if self.should_use_rag(intent):
            return "rag"

        if intent == "market":
            return "market_prediction"

        return "rag"