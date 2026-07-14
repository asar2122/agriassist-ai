class IntentClassifier:

    def __init__(self):

        self.intent_keywords = {

            "crop": [
                "crop",
                "cultivation",
                "grow",
                "plant",
                "seed",
                "harvest",
            ],

            "disease": [
                "disease",
                "leaf",
                "infection",
                "fungus",
                "blight",
                "spot",
            ],

            "fertilizer": [
                "fertilizer",
                "nitrogen",
                "phosphorus",
                "potassium",
                "urea",
                "dap",
                "npk",
            ],

            "pest": [
                "pest",
                "insect",
                "worm",
                "borer",
                "aphid",
            ],

            "market": [
                "price",
                "market",
                "cost",
                "rate",
            ],
        }


    def predict(self, text):

        text = text.lower()

        scores = {}

        for intent, keywords in self.intent_keywords.items():

            score = 0

            for keyword in keywords:

                if keyword in text:

                    score += 1

            scores[intent] = score


        best_intent = max(
            scores,
            key=scores.get,
        )


        if scores[best_intent] == 0:

            return "general"


        return best_intent