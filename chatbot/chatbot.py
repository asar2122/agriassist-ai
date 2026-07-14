from chatbot.intent_classifier import IntentClassifier
from chatbot.llm_client import LLMClient
from chatbot.prompt_builder import PromptBuilder
from chatbot.query_router import QueryRouter

from rag.retriever import AgricultureRetriever


class AgriChatbot:

    def __init__(self):

        self.intent_classifier = IntentClassifier()

        self.query_router = QueryRouter()

        self.retriever = AgricultureRetriever()

        self.llm_client = LLMClient()

    def get_response(self, user_question):

        if not user_question.strip():

            return {
                "answer": (
                    "Please enter an agriculture question."
                ),
                "sources": [],
                "intent": "general",
            }

        intent = self.intent_classifier.predict(
            user_question
        )

        route = self.query_router.route(
            intent
        )

        if route == "rag":

            return self._rag_response(
                question=user_question,
                intent=intent,
            )

        if route == "market_prediction":

            return {
                "answer": (
                    "Market prediction will be added "
                    "in a later phase."
                ),
                "sources": [],
                "intent": intent,
            }

        return {
            "answer": (
                "Unable to process the question."
            ),
            "sources": [],
            "intent": intent,
        }

    def _rag_response(
        self,
        question,
        intent,
    ):

        results = self.retriever.retrieve(
            question=question,
            top_k=3,
            minimum_score=0.40,
        )

        if not results:

            return {
                "answer": (
                    "I could not find sufficient "
                    "information in the local "
                    "agriculture knowledge base."
                ),
                "sources": [],
                "intent": intent,
            }

        context_parts = []

        for result in results:

            context_parts.append(
                result["text"]
            )

        context = "\n\n".join(
            context_parts
        )

        prompt = PromptBuilder.build_prompt(
            question=question,
            context=context,
        )

        answer = self.llm_client.generate(
            prompt
        )

        sources = []

        for result in results:

            source = {
                "file_name": result["file_name"],
                "page": result.get("page"),
                "score": result["score"],
            }

            if source not in sources:

                sources.append(
                    source
                )

        return {
            "answer": answer,
            "sources": sources,
            "intent": intent,
        }