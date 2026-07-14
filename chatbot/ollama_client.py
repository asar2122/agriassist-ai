import json
import urllib.error
import urllib.request


class OllamaClient:
    def __init__(
        self,
        model_name="llama3.2:3b",
        base_url="http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt):
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            },
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:

                result = json.loads(
                    response.read().decode("utf-8")
                )

                return result.get(
                    "response",
                    "No response generated.",
                ).strip()

        except urllib.error.URLError:
            return (
                "Unable to connect to Ollama. "
                "Make sure Ollama is running."
            )

        except Exception as error:
            return f"Ollama error: {error}"