from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from openpi.rl.http_protocol import SampleRequest, RolloutUpdateRequest


@dataclass
class OpenPIServerClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 600.0

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def sample(self, observation: Any, mode: str = "train", return_values: bool = True) -> dict[str, Any]:
        return self._request("POST", "/sample", json=SampleRequest(observation=observation, mode=mode, return_values=return_values).to_dict())

    def update(self, rollout: RolloutUpdateRequest) -> dict[str, Any]:
        return self._request("POST", "/update", json=rollout.to_dict())

    def _request(self, method: str, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.request(method, self.base_url + path, json=json, timeout=self.timeout)
        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise requests.HTTPError(
                f"{response.status_code} Server Error for {response.url}: {detail}", response=response
            )
        return response.json()
