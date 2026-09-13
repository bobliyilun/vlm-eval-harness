"""Small HTTP inference adapter building blocks."""

import json
from typing import Callable, Optional
from urllib.request import Request, urlopen


class HTTPInferenceAdapter:
    """Send a JSON request and turn a JSON response into one prediction."""

    def __init__(
        self,
        url: str,
        build_request: Callable[[dict], dict],
        parse_response: Callable[[dict], object],
        *,
        timeout: float = 30,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.url = url
        self.build_request = build_request
        self.parse_response = parse_response
        self.timeout = timeout
        self.headers = headers or {}

    def infer(self, record: dict) -> object:
        """Return the parsed prediction for one dataset record."""
        request = Request(
            self.url,
            data=json.dumps(self.build_request(record)).encode(),
            headers={
                "Content-Type": "application/json", "Accept": "application/json", **self.headers
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return self.parse_response(json.load(response))


class OpenAICompatibleAdapter(HTTPInferenceAdapter):
    """Send a text prompt to an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self, url: str, model: str, *, api_key: Optional[str] = None, timeout: float = 30
    ) -> None:
        super().__init__(
            url,
            lambda record: {
                "model": model,
                "messages": [{"role": "user", "content": record["prompt"]}],
            },
            lambda response: response["choices"][0]["message"]["content"],
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        )
