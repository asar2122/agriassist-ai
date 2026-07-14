import json
import socket
import urllib.error
import urllib.request


class OllamaClient:

    def __init__(
        self,
        model_name="llama3.2:3b",
        base_url="http://localhost:11434",
    ):

        self.model_name = model_name

        self.base_url = base_url.rstrip("/")

    def generate(self, prompt):

        # Validate prompt

        if not prompt or not prompt.strip():

            return (
                "Please provide a valid "
                "agriculture question."
            )

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }

        data = json.dumps(
            payload
        ).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:

                response_data = response.read()

                result = json.loads(
                    response_data.decode("utf-8")
                )

                generated_response = result.get(
                    "response",
                    "",
                ).strip()

                if not generated_response:

                    return (
                        "The language model did not "
                        "generate a response."
                    )

                return generated_response

        except urllib.error.HTTPError as error:

            return (
                "Ollama returned an HTTP error: "
                f"{error.code}"
            )

        except urllib.error.URLError:

            return (
                "The local AI language model is "
                "currently unavailable. "
                "Please make sure Ollama is running."
            )

        except socket.timeout:

            return (
                "The AI model took too long to respond. "
                "Please try again."
            )

        except json.JSONDecodeError:

            return (
                "The AI model returned an invalid "
                "response."
            )

        except Exception as error:

            return (
                "An unexpected Ollama error occurred: "
                f"{error}"
            )