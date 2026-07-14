import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

class LLMClient:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model_name = "llama-3.1-8b-instant"

    def generate(self, prompt):

        if not prompt or not prompt.strip():
            return (
                "Please provide a valid "
                "agriculture question."
            )

        try:

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.2,
            )

            generated_response = (
                response.choices[0]
                .message.content
                .strip()
            )

            if not generated_response:
                return (
                    "The language model did not "
                    "generate a response."
                )

            return generated_response

        except Exception as error:

            return (
                "The cloud AI language model is "
                "currently unavailable. "
                f"Error: {error}"
            )