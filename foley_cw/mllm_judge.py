"""Qwen-Omni MLLM judge for the validity calibration sidecar (foley_cw/mllm_judge.py).

Serves manual §3.3 (experiment/LONG_RANGE_EXPERIMENT_PLAN.md, Phase 0.5 reliability
gate): "MLLM probes at temperature 0 with a test-retest subset and versioned prompts",
and the §"MLLM probe budget & prompt versioning" rule (fixed prompts, temperature 0,
response caching, hard call budget).

Contract:
  * ``QwenOmniJudge.judge(wav_path, axis)`` -> SelfTarget(kind=CATEGORICAL) for the
    three sidecar axes:
      presence  {"present": bool}       -> label "present" / "absent"
      class     {"coarse_class": str}   -> the class string (validated against
                                           ``coarse_classes`` when provided)
      timing    {"onset_s": number}     -> label = int(floor(onset_s / timing_bin_s))
  * temperature is ALWAYS 0; prompts are versioned files
    configs/mllm_prompts/<axis.id>__<prompt_version>.txt.  ``model``,
    ``prompt_version``, ``axis.id``, and sha256(wav bytes) form the cache key, so any
    prompt/model bump busts the cache; cache hits make NO network call and consume NO
    budget.
  * Every real network-call cycle is (a) pre-charged against a persistent budget
    (cache_dir/budget.json; raises MLLMBudgetExceeded when the charge would exceed
    ``budget_max_calls``) and (b) journaled to cache_dir/calls.jsonl.  A parse-retry is
    a second cycle and is charged separately.
  * ``test_retest(wav_paths, axis, n)`` re-judges with cache-key suffix "|retest<i>",
    so the retest copies BYPASS the primary cache (each retest copy is still cached
    under its own suffixed key, making a re-run of the retest itself free).

Network notes (login-node specifics; LOAD-BEARING):
  * ``httpx.Client(trust_env=False, ...)``: the login node exports
    http_proxy/https_proxy env vars (a localhost:7890 proxy) that must NOT capture
    this host; the only proxy ever used is the explicit ``proxy`` argument.
  * When a ``transport`` is injected (tests use httpx.MockTransport), ``proxy`` is
    DISABLED: httpx mounts proxy transports with precedence over the default
    transport, so keeping the proxy would route mock-intended requests to the real
    network.
  * The request URL is ``base_url.rstrip('/') + '/v1/chat/completions'`` where
    base_url defaults to the ``openAiCompatible`` value of the key CSV.  Pass
    ``base_url`` (or reassign ``.request_url`` after construction) once a live smoke
    confirms the exact compatible-mode path.

httpx is the only non-numpy dependency and is guarded at import: this module stays
importable on a numpy-only environment; constructing QwenOmniJudge without httpx
raises ImportError with the dependency named.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .agreement import agreement
from .types import Axis, AxisKind, SelfTarget

try:  # guarded heavy/optional dep — numpy core must import without it
    import httpx
except ImportError as _exc:  # pragma: no cover - exercised only on minimal envs
    httpx = None  # type: ignore[assignment]
    _HTTPX_IMPORT_ERROR: Optional[ImportError] = _exc
else:
    _HTTPX_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Default key file (2-column CSV) outside the repo; override in tests.
_DEFAULT_KEY_CSV: str = "/HOME/paratera_xy/pxy1289/HDD_POOL/HaocunYe/qwen-api.csv"
#: Cluster forward proxy that reaches the API host (the env proxies do NOT).
_DEFAULT_PROXY: str = "http://172.16.31.200:3138"
#: Short per-call user instruction; the rating rubric lives in the system prompt file.
_USER_INSTRUCTION: str = (
    "Rate this Foley audio clip according to the system instructions "
    "and return the JSON object."
)
#: Appended to the user text on the single parse-retry.
_PARSE_RETRY_SUFFIX: str = " Return ONLY the JSON object."


class MLLMBudgetExceeded(Exception):
    """Raised when a real MLLM call would exceed the persistent call budget."""


def read_key_csv(path: str | Path) -> dict[str, str]:
    """Parse a 2-column key CSV of ``key,value`` rows into a dict.

    Handles a BOM-prefixed first line (e.g. '\\ufeffid,5573406'), skips blank lines
    and lines without a comma; the value may itself contain commas (split on the
    first comma only).
    """
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or "," not in line:
                continue
            key, value = line.split(",", 1)
            out[key.strip()] = value.strip()
    return out


class QwenOmniJudge:
    """Qwen-Omni chat-completions judge with cache, budget, and journaling.

    Parameters
    ----------
    key_csv:
        Path to the 2-column credentials CSV (apiKey / apiHost / openAiCompatible).
    model:
        Chat-completions model name.
    proxy:
        Explicit forward proxy URL, or None for a direct connection.  Ignored when
        ``transport`` is given (see module docstring).
    cache_dir:
        Directory for response cache files, budget.json, and calls.jsonl.
    prompt_dir / prompt_version:
        System prompts are read from ``prompt_dir/<axis.id>__<prompt_version>.txt``.
    timeout_s / max_retries / backoff_s:
        HTTP timeout; max POST attempts per call cycle; exponential backoff base
        (sleep = backoff_s * 2**attempt after a 5xx/transport failure).
    budget_max_calls:
        Hard cap on real call cycles, persisted across instances in budget.json.
    base_url:
        Overrides the CSV ``openAiCompatible`` value.  The request URL is
        ``base_url.rstrip('/') + '/v1/chat/completions'`` and is stored on
        ``self.request_url`` (reassignable) so the exact path can be corrected after
        the live smoke without a code change.
    transport:
        Optional httpx transport (tests inject httpx.MockTransport).
    coarse_classes:
        Closed label set for the class axis; fills ``{CLASSES}`` in the prompt
        template and validates the returned label.
    timing_bin_s:
        Bin width (seconds) for quantising the timing axis onset.
    """

    def __init__(
        self,
        key_csv: str = _DEFAULT_KEY_CSV,
        model: str = "qwen3.5-omni-plus",
        proxy: str | None = _DEFAULT_PROXY,
        cache_dir: Path = _REPO_ROOT / "results" / "mllm_cache",
        prompt_dir: Path = _REPO_ROOT / "configs" / "mllm_prompts",
        prompt_version: str = "v1",
        timeout_s: float = 120,
        max_retries: int = 5,
        backoff_s: float = 2.0,
        budget_max_calls: int = 500,
        base_url: str | None = None,
        transport: Any = None,
        coarse_classes: Optional[list[str]] = None,
        timing_bin_s: float = 0.5,
    ) -> None:
        if httpx is None:  # pragma: no cover - minimal envs only
            raise ImportError(
                "foley_cw.mllm_judge.QwenOmniJudge requires httpx; "
                "install it (pip install httpx) — the numpy core does not need it."
            ) from _HTTPX_IMPORT_ERROR
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        keys = read_key_csv(key_csv)
        if "apiKey" not in keys:
            raise ValueError(f"key CSV {key_csv!r} has no 'apiKey' row")
        self.api_key: str = keys["apiKey"]

        if base_url is None:
            base_url = keys.get("openAiCompatible")
            if not base_url:
                raise ValueError(
                    f"key CSV {key_csv!r} has no 'openAiCompatible' row and no "
                    "base_url was given"
                )
        self.base_url: str = base_url
        # Reassignable: the compatible-mode path is confirmed by a live smoke later.
        # The key CSV's openAiCompatible value already ends in /v1 (Aliyun MaaS
        # compatible-mode); avoid doubling the version segment.
        root = self.base_url.rstrip("/")
        suffix = "/chat/completions" if root.endswith("/v1") else "/v1/chat/completions"
        self.request_url: str = root + suffix

        self.model = model
        self.prompt_dir = Path(prompt_dir)
        self.prompt_version = prompt_version
        self.max_retries = int(max_retries)
        self.backoff_s = float(backoff_s)
        self.budget_max_calls = int(budget_max_calls)
        self.coarse_classes = list(coarse_classes) if coarse_classes else None
        self.timing_bin_s = float(timing_bin_s)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._budget_path = self.cache_dir / "budget.json"
        self._journal_path = self.cache_dir / "calls.jsonl"

        # An injected transport must own ALL traffic: httpx mounts proxy transports
        # with precedence over the default transport, so the proxy is disabled here.
        effective_proxy = None if transport is not None else proxy
        # trust_env=False is LOAD-BEARING: the login node's http_proxy/https_proxy
        # env vars (localhost:7890) must not capture this host.
        # verify=False is required through the designated qwen proxy
        # (172.16.31.200:3138): it terminates/re-signs TLS, so the presented
        # certificate does not match the aliyuncs hostname. The API key still
        # authenticates the session; the route is an internal HPC egress proxy
        # chosen by the user. Recorded as an accepted risk in the run journal.
        self._client = httpx.Client(
            trust_env=False,
            proxy=effective_proxy,
            timeout=timeout_s,
            transport=transport,
            verify=transport is not None,  # MockTransport tests keep verify on
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def judge(self, wav_path: Path, axis: Axis, cache_suffix: str = "") -> SelfTarget:
        """Judge one wav on one axis; cached, budgeted, temperature-0.

        ``cache_suffix`` is appended to the cache-key material before hashing;
        ``test_retest`` uses it to bypass the primary cache entry.
        """
        wav_bytes = Path(wav_path).read_bytes()
        cache_key = self._cache_key(axis, wav_bytes, cache_suffix)
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            record = json.loads(cache_file.read_text(encoding="utf-8"))
            payload = self._extract_json(record["content"])
            return self._json_to_target(payload, axis)

        system_prompt = self._load_prompt(axis)
        content = self._call(system_prompt, _USER_INSTRUCTION, wav_bytes, axis,
                             cache_key, wav_path)
        try:
            payload = self._extract_json(content)
        except ValueError:
            # One parse-retry; schema violations (valid JSON, wrong shape) do NOT
            # retry — they indicate a prompt bug, not a formatting slip.
            content = self._call(
                system_prompt, _USER_INSTRUCTION + _PARSE_RETRY_SUFFIX,
                wav_bytes, axis, cache_key, wav_path,
            )
            payload = self._extract_json(content)
        target = self._json_to_target(payload, axis)

        # Cache only after a successful parse so bad responses never poison it.
        record = {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "axis_id": axis.id,
            "wav_sha256": hashlib.sha256(wav_bytes).hexdigest(),
            "cache_suffix": cache_suffix,
            "content": content,
            "label": target.label,
        }
        cache_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return target

    def test_retest(self, wav_paths: list[Path], axis: Axis, n: int = 2) -> float:
        """Agreement of n repeated judge calls per wav, primary cache bypassed.

        Each repeat i uses cache-key suffix '|retest<i>' so it cannot hit the primary
        cache entry (or another repeat's).  Agreement per wav uses the axis's
        registered metric; the mean over wavs is returned (NaN for an empty list).
        """
        per_wav: list[float] = []
        for wav in wav_paths:
            targets = [
                self.judge(wav, axis, cache_suffix=f"|retest{i}") for i in range(n)
            ]
            per_wav.append(agreement(targets, axis.agreement))
        if not per_wav:
            return float("nan")
        return float(np.mean(per_wav))

    # ------------------------------------------------------------------
    # Prompt / cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, axis: Axis, wav_bytes: bytes, cache_suffix: str) -> str:
        wav_hash = hashlib.sha256(wav_bytes).hexdigest()
        material = "|".join([self.model, self.prompt_version, axis.id, wav_hash])
        return hashlib.sha256((material + cache_suffix).encode("utf-8")).hexdigest()

    def _load_prompt(self, axis: Axis) -> str:
        path = self.prompt_dir / f"{axis.id}__{self.prompt_version}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"MLLM prompt file not found: {path} "
                f"(axis {axis.id!r}, prompt_version {self.prompt_version!r})"
            )
        text = path.read_text(encoding="utf-8")
        # str.replace, NOT str.format: the templates contain literal JSON braces.
        if "{CLASSES}" in text:
            if not self.coarse_classes:
                raise ValueError(
                    f"prompt {path.name} contains {{CLASSES}} but no coarse_classes "
                    "were given to QwenOmniJudge"
                )
            text = text.replace("{CLASSES}", ", ".join(self.coarse_classes))
        return text

    # ------------------------------------------------------------------
    # Budget / journal
    # ------------------------------------------------------------------

    def _charge_budget(self) -> int:
        """Pre-charge one real call cycle; raise if it would exceed the budget."""
        calls = 0
        if self._budget_path.exists():
            try:
                calls = int(json.loads(
                    self._budget_path.read_text(encoding="utf-8")
                ).get("calls", 0))
            except (ValueError, json.JSONDecodeError):
                calls = 0
        if calls + 1 > self.budget_max_calls:
            raise MLLMBudgetExceeded(
                f"MLLM call budget exhausted: {calls} calls recorded in "
                f"{self._budget_path}, budget_max_calls={self.budget_max_calls}"
            )
        calls += 1
        self._budget_path.write_text(json.dumps({"calls": calls}), encoding="utf-8")
        return calls

    def _journal(self, axis: Axis, cache_key: str, wav_path: Path,
                 status: int, attempts: int) -> None:
        entry = {
            "ts": time.time(),
            "model": self.model,
            "prompt_version": self.prompt_version,
            "axis_id": axis.id,
            "cache_key": cache_key,
            "wav": str(wav_path),
            "status": status,
            "attempts": attempts,
        }
        with open(self._journal_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Network call cycle (budgeted, retried, journaled)
    # ------------------------------------------------------------------

    def _call(self, system_prompt: str, user_text: str, wav_bytes: bytes,
              axis: Axis, cache_key: str, wav_path: Path) -> str:
        """One budgeted call cycle: POST with 5xx/transport retries; return content."""
        self._charge_budget()
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                # DashScope compatible-mode parses 'data' as a URL;
                                # bare base64 is rejected ("URL does not appear to be
                                # valid") — the data-URI prefix is required.
                                "data": "data:audio/wav;base64,"
                                        + base64.b64encode(wav_bytes).decode("ascii"),
                                "format": "wav",
                            },
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 256,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        last_err = "no attempt made"
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(self.request_url, json=body, headers=headers)
            except httpx.TransportError as exc:  # includes timeouts
                last_err = f"transport error: {exc!r}"
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_s * (2 ** attempt))
                continue
            if resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_s * (2 ** attempt))
                continue
            if resp.status_code != 200:
                self._journal(axis, cache_key, wav_path,
                              status=resp.status_code, attempts=attempt + 1)
                raise RuntimeError(
                    f"MLLM call failed (HTTP {resp.status_code}, no retry on 4xx): "
                    f"{resp.text[:500]}"
                )
            try:
                data = resp.json()
            except (json.JSONDecodeError, ValueError) as exc:
                # Charged call: journal it even though the body is unusable.
                self._journal(axis, cache_key, wav_path, status=200,
                              attempts=attempt + 1)
                raise RuntimeError(
                    f"MLLM response body is not JSON: {resp.text[:500]}"
                ) from exc
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                self._journal(axis, cache_key, wav_path, status=200,
                              attempts=attempt + 1)
                raise RuntimeError(
                    f"MLLM response missing choices[0].message.content: "
                    f"{json.dumps(data)[:500]}"
                ) from exc
            self._journal(axis, cache_key, wav_path, status=200,
                          attempts=attempt + 1)
            return str(content)

        self._journal(axis, cache_key, wav_path, status=-1,
                      attempts=self.max_retries)
        raise RuntimeError(
            f"MLLM call failed after {self.max_retries} attempts; "
            f"last error: {last_err}"
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(content: str) -> dict:
        """Strict JSON object from assistant content; markdown fences stripped.

        Raises ValueError when no JSON object can be parsed (this is the only
        condition that triggers the single parse-retry in ``judge``).
        """
        text = content.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            i, j = text.find("{"), text.rfind("}")
            payload = None
            if i != -1 and j > i:
                try:
                    payload = json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    payload = None
            if payload is None:
                raise ValueError(
                    f"MLLM content is not valid JSON: {content[:200]!r}"
                ) from None
        if not isinstance(payload, dict):
            raise ValueError(f"MLLM content is not a JSON object: {content[:200]!r}")
        return payload

    def _json_to_target(self, payload: dict, axis: Axis) -> SelfTarget:
        """Per-axis schema -> categorical SelfTarget; strict types, no coercion."""
        if axis.id == "presence":
            if "present" not in payload or not isinstance(payload["present"], bool):
                raise ValueError(
                    f"presence schema requires boolean 'present'; got {payload!r}"
                )
            label: Any = "present" if payload["present"] else "absent"
        elif axis.id == "class":
            if "coarse_class" not in payload or not isinstance(payload["coarse_class"], str):
                raise ValueError(
                    f"class schema requires string 'coarse_class'; got {payload!r}"
                )
            label = payload["coarse_class"]
            if self.coarse_classes is not None and label not in self.coarse_classes:
                raise ValueError(
                    f"coarse_class {label!r} not in the registered class list "
                    f"{self.coarse_classes!r}"
                )
        elif axis.id == "timing":
            onset = payload.get("onset_s")
            if isinstance(onset, bool) or not isinstance(onset, (int, float)):
                raise ValueError(
                    f"timing schema requires numeric 'onset_s'; got {payload!r}"
                )
            label = int(math.floor(float(onset) / self.timing_bin_s))
        else:
            raise ValueError(
                f"QwenOmniJudge has no response schema for axis {axis.id!r} "
                "(supported: presence, class, timing)"
            )
        return SelfTarget(axis_id=axis.id, kind=AxisKind.CATEGORICAL, label=label)
