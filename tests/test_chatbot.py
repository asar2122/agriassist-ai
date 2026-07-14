from chatbot.intent_classifier import IntentClassifier
from chatbot.chatbot import AgriChatbot


# --------------------------------------------------
# INTENT CLASSIFIER TESTS
# --------------------------------------------------


def test_crop_intent():

    classifier = IntentClassifier()

    result = classifier.predict(
        "How to cultivate rice crop?"
    )

    assert result == "crop"


def test_disease_intent():

    classifier = IntentClassifier()

    result = classifier.predict(
        "My tomato leaf has disease"
    )

    assert result == "disease"


def test_fertilizer_intent():

    classifier = IntentClassifier()

    result = classifier.predict(
        "Which fertilizer contains nitrogen?"
    )

    assert result == "fertilizer"


def test_general_intent():

    classifier = IntentClassifier()

    result = classifier.predict(
        "Hello"
    )

    assert result == "general"


# --------------------------------------------------
# CHATBOT / RAG TESTS
# --------------------------------------------------


def test_relevant_agriculture_question():

    chatbot = AgriChatbot()

    response = chatbot.get_response(
        "What are the symptoms of rice blast?"
    )

    assert response is not None

    assert isinstance(
        response,
        dict,
    )

    assert "answer" in response

    assert "sources" in response

    assert "intent" in response

    assert isinstance(
        response["answer"],
        str,
    )

    assert len(
        response["answer"]
    ) > 0

    assert isinstance(
        response["sources"],
        list,
    )

    assert len(
        response["sources"]
    ) > 0


def test_irrelevant_question_rejection():

    chatbot = AgriChatbot()

    response = chatbot.get_response(
        "How do I repair a motorcycle?"
    )

    assert response is not None

    assert isinstance(
        response,
        dict,
    )

    assert "answer" in response

    assert "sources" in response

    assert "intent" in response

    assert response["sources"] == []

    assert (
        "could not find sufficient information"
        in response["answer"].lower()
    )


def test_empty_question():

    chatbot = AgriChatbot()

    response = chatbot.get_response("")

    assert response is not None

    assert isinstance(
        response,
        dict,
    )

    assert "answer" in response

    assert "sources" in response

    assert "intent" in response

    assert response["sources"] == []

    assert (
        response["answer"]
        == "Please enter an agriculture question."
    )

    assert response["intent"] == "general"