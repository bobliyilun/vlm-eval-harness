"""Small HTTP inference adapter building blocks."""

import json
from typing import Callable
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
    ) -> None:
        self.url = url
        self.build_request = build_request
        self.parse_response = parse_response
        self.timeout = timeout

    def infer(self, record: dict) -> object:
        """Return the parsed prediction for one dataset record."""
        request = Request(
            self.url,
            data=json.dumps(self.build_request(record)).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return self.parse_response(json.load(response))
