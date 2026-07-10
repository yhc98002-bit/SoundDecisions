"""Tests for foley_cw/mllm_judge.py — Qwen-Omni MLLM judge (manual §3.3 sidecar).

CPU-only, NO network, NO GPU: every HTTP exchange goes through httpx.MockTransport
(the judge disables its proxy whenever a transport is injected, so nothing can
escape to the real network).  Deterministic: wav fixtures use seeded rngs; canned
responses are fixed strings.

Key contracts checked here:
  * key-CSV parsing including the BOM-prefixed first line.
  * request body: temperature == 0, max_tokens == 256, input_audio part with the
    base64 wav bytes, versioned system prompt, Bearer auth, /v1/chat/completions URL.
  * cache: miss -> hit (second call makes 0 requests); prompt_version busts the cache.
  * retries: 5xx then success; one parse-retry on non-JSON content.
  * budget: persistent, raises MLLMBudgetExceeded; cache hits cost nothing.
  * per-axis schemas: presence/class/timing -> categorical SelfTargets.
  * test_retest bypasses the primary cache via '|retest<i>' suffixed keys.
"""

from __future__ import annotations

import base64
import json
import wave
from pathlib import Path

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")

from foley_cw.mllm_judge import (  # noqa: E402
    MLLMBudgetExceeded,
    QwenOmniJudge,
    read_key_csv,
)
from foley_cw.types import (  # noqa: E402
    AgreementMetric,
    Axis,
    AxisKind,
    AxisTier,
    SelfTarget,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = REPO_ROOT / "configs" / "mllm_prompts"


# --------------------------------------------------------------------------------------
# Helpers / fixtures
# --------------------------------------------------------------------------------------

def _write_wav(path: Path, seed: int = 0) -> Path:
    """Write a tiny deterministic 16 kHz mono PCM16 wav (stdlib wave only)."""
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(160) * 1000.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())
    return path


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _capture_handler(requests: list, content: str = '{"present": true}'):
    """MockTransport handler returning a fixed payload; records (request, body)."""
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request, json.loads(request.read().decode("utf-8"))))
        return httpx.Response(200, json=_chat_response(content))
    return handler


@pytest.fixture
def key_csv(tmp_path):
    p = tmp_path / "qwen-api.csv"
    p.write_text(
        "\ufeffid,5573406\n"
        "apiKey,sk-test-key\n"
        "apiHost,llm-host.aliyuncs.com\n"
        "openAiCompatible,https://llm-host.aliyuncs.com/compatible\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture
def wav_file(tmp_path):
    return _write_wav(tmp_path / "clip0.wav", seed=0)


@pytest.fixture
def presence_axis():
    return Axis(
        id="presence",
        name="event-sound presence",
        tier=AxisTier.TIER1,
        kind=AxisKind.CATEGORICAL,
        agreement=AgreementMetric.EXACT_MATCH,
        measure="presence_detector",
    )


@pytest.fixture
def class_axis():
    return Axis(
        id="class",
        name="coarse event class",
        tier=AxisTier.TIER1,
        kind=AxisKind.CATEGORICAL,
        agreement=AgreementMetric.KRIPPENDORFF_ALPHA,
        measure="audio_tagger_top1",
    )


@pytest.fixture
def timing_axis():
    return Axis(
        id="timing",
        name="gross timing",
        tier=AxisTier.TIER1,
        kind=AxisKind.CATEGORICAL,
        agreement=AgreementMetric.EXACT_MATCH,
        measure="onset_timing_bin",
    )


def make_judge(key_csv: str, tmp_path: Path, handler, **kwargs) -> QwenOmniJudge:
    """Judge wired to a MockTransport and a tmp cache dir (never the repo cache)."""
    defaults = dict(
        key_csv=key_csv,
        cache_dir=tmp_path / "mllm_cache",
        prompt_dir=PROMPT_DIR,
        backoff_s=0.0,
        transport=httpx.MockTransport(handler),
    )
    defaults.update(kwargs)
    return QwenOmniJudge(**defaults)


# --------------------------------------------------------------------------------------
# read_key_csv
# --------------------------------------------------------------------------------------

class TestReadKeyCsv:
    def test_parses_pairs_and_strips_bom(self, key_csv):
        keys = read_key_csv(key_csv)
        assert keys["id"] == "5573406", f"BOM not stripped: {list(keys)!r}"
        assert "\ufeffid" not in keys
        assert keys["apiKey"] == "sk-test-key"
        assert keys["apiHost"] == "llm-host.aliyuncs.com"
        assert keys["openAiCompatible"] == "https://llm-host.aliyuncs.com/compatible"

    def test_skips_blank_and_commaless_lines(self, tmp_path):
        p = tmp_path / "k.csv"
        p.write_text("apiKey,sk-x\n\nnotapair\nopenAiCompatible,https://h/compat\n",
                     encoding="utf-8")
        keys = read_key_csv(p)
        assert keys == {"apiKey": "sk-x", "openAiCompatible": "https://h/compat"}

    def test_value_may_contain_commas(self, tmp_path):
        p = tmp_path / "k.csv"
        p.write_text("apiKey,sk-a,b,c\n", encoding="utf-8")
        assert read_key_csv(p)["apiKey"] == "sk-a,b,c"


# --------------------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------------------

class TestJudgeRequest:
    def test_temperature_zero_and_input_audio_part(self, key_csv, tmp_path,
                                                   wav_file, presence_axis):
        requests: list = []
        judge = make_judge(key_csv, tmp_path, _capture_handler(requests))
        target = judge.judge(wav_file, presence_axis)

        assert len(requests) == 1
        _request, body = requests[0]
        assert body["temperature"] == 0
        assert body["max_tokens"] == 256
        assert body["model"] == "qwen3.5-omni-plus"

        assert body["messages"][0]["role"] == "system"
        user = body["messages"][1]
        assert user["role"] == "user"
        audio_parts = [p for p in user["content"] if p.get("type") == "input_audio"]
        assert len(audio_parts) == 1, "user content must carry one input_audio part"
        data = audio_parts[0]["input_audio"]["data"]
        assert audio_parts[0]["input_audio"]["format"] == "wav"
        # DashScope compatible-mode parses 'data' as a URL: the data-URI prefix is
        # mandatory (live-validated 2026-06-11; bare base64 -> HTTP 400).
        prefix = "data:audio/wav;base64,"
        assert data.startswith(prefix)
        assert base64.b64decode(data[len(prefix):]) == wav_file.read_bytes()

        assert isinstance(target, SelfTarget)
        assert target.kind is AxisKind.CATEGORICAL
        assert target.label == "present"

    def test_url_and_auth_header(self, key_csv, tmp_path, wav_file, presence_axis):
        requests: list = []
        judge = make_judge(key_csv, tmp_path, _capture_handler(requests))
        judge.judge(wav_file, presence_axis)
        request, _body = requests[0]
        assert str(request.url) == (
            "https://llm-host.aliyuncs.com/compatible/v1/chat/completions"
        )
        assert request.headers["Authorization"] == "Bearer sk-test-key"

    def test_base_url_override(self, key_csv, tmp_path, wav_file, presence_axis):
        requests: list = []
        judge = make_judge(key_csv, tmp_path, _capture_handler(requests),
                           base_url="https://other-host/api/")
        judge.judge(wav_file, presence_axis)
        request, _body = requests[0]
        assert str(request.url) == "https://other-host/api/v1/chat/completions"

    def test_absent_label(self, key_csv, tmp_path, wav_file, presence_axis):
        judge = make_judge(key_csv, tmp_path,
                           _capture_handler([], content='{"present": false}'))
        assert judge.judge(wav_file, presence_axis).label == "absent"

    def test_transport_overrides_default_proxy(self, key_csv, tmp_path, wav_file,
                                               presence_axis):
        """With an injected transport the default proxy must be neutralised —
        otherwise httpx would mount a real proxy transport over the mock."""
        requests: list = []
        # NOTE: proxy is left at its non-None default on purpose.
        judge = QwenOmniJudge(
            key_csv=key_csv,
            cache_dir=tmp_path / "mllm_cache",
            prompt_dir=PROMPT_DIR,
            backoff_s=0.0,
            transport=httpx.MockTransport(_capture_handler(requests)),
        )
        assert judge.judge(wav_file, presence_axis).label == "present"
        assert len(requests) == 1


# --------------------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------------------

class TestCache:
    def test_miss_then_hit_makes_zero_requests(self, key_csv, tmp_path, wav_file,
                                               presence_axis):
        requests: list = []
        judge = make_judge(key_csv, tmp_path, _capture_handler(requests))
        first = judge.judge(wav_file, presence_axis)
        assert len(requests) == 1
        second = judge.judge(wav_file, presence_axis)
        assert len(requests) == 1, "cache hit must make no network request"
        assert second.label == first.label == "present"

    def test_prompt_version_busts_cache(self, key_csv, tmp_path, wav_file,
                                        presence_axis):
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        v1_text = (PROMPT_DIR / "presence__v1.txt").read_text(encoding="utf-8")
        (prompt_dir / "presence__v1.txt").write_text(v1_text, encoding="utf-8")
        (prompt_dir / "presence__v2.txt").write_text(v1_text, encoding="utf-8")

        requests: list = []
        cache_dir = tmp_path / "mllm_cache"
        judge_v1 = make_judge(key_csv, tmp_path, _capture_handler(requests),
                              prompt_dir=prompt_dir, cache_dir=cache_dir)
        judge_v1.judge(wav_file, presence_axis)
        assert len(requests) == 1

        judge_v2 = make_judge(key_csv, tmp_path, _capture_handler(requests),
                              prompt_dir=prompt_dir, cache_dir=cache_dir,
                              prompt_version="v2")
        judge_v2.judge(wav_file, presence_axis)
        assert len(requests) == 2, "new prompt_version must bypass the v1 cache entry"

    def test_different_wav_busts_cache(self, key_csv, tmp_path, presence_axis):
        requests: list = []
        judge = make_judge(key_csv, tmp_path, _capture_handler(requests))
        judge.judge(_write_wav(tmp_path / "a.wav", seed=1), presence_axis)
        judge.judge(_write_wav(tmp_path / "b.wav", seed=2), presence_axis)
        assert len(requests) == 2

    def test_cache_shared_across_instances(self, key_csv, tmp_path, wav_file,
                                           presence_axis):
        requests: list = []
        cache_dir = tmp_path / "mllm_cache"
        make_judge(key_csv, tmp_path, _capture_handler(requests),
                   cache_dir=cache_dir).judge(wav_file, presence_axis)
        make_judge(key_csv, tmp_path, _capture_handler(requests),
                   cache_dir=cache_dir).judge(wav_file, presence_axis)
        assert len(requests) == 1


# --------------------------------------------------------------------------------------
# Retries and parse-retry
# --------------------------------------------------------------------------------------

class TestRetries:
    def test_5xx_then_success(self, key_csv, tmp_path, wav_file, presence_axis):
        calls: list = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(503, text="upstream busy")
            return httpx.Response(200, json=_chat_response('{"present": true}'))

        judge = make_judge(key_csv, tmp_path, handler)  # backoff_s=0 in make_judge
        target = judge.judge(wav_file, presence_axis)
        assert target.label == "present"
        assert len(calls) == 2, "one 5xx retry expected"

    def test_exhausted_5xx_raises(self, key_csv, tmp_path, wav_file, presence_axis):
        calls: list = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(500, text="boom")

        judge = make_judge(key_csv, tmp_path, handler, max_retries=3)
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            judge.judge(wav_file, presence_axis)
        assert len(calls) == 3

    def test_4xx_raises_without_retry(self, key_csv, tmp_path, wav_file,
                                      presence_axis):
        calls: list = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(401, text="bad key")

        judge = make_judge(key_csv, tmp_path, handler)
        with pytest.raises(RuntimeError, match="HTTP 401"):
            judge.judge(wav_file, presence_axis)
        assert len(calls) == 1

    def test_parse_retry_on_non_json_then_valid(self, key_csv, tmp_path, wav_file,
                                                presence_axis):
        bodies: list = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.read().decode("utf-8")))
            if len(bodies) == 1:
                return httpx.Response(
                    200, json=_chat_response("I think the sound is present.")
                )
            return httpx.Response(200, json=_chat_response('{"present": true}'))

        judge = make_judge(key_csv, tmp_path, handler)
        target = judge.judge(wav_file, presence_axis)
        assert target.label == "present"
        assert len(bodies) == 2, "exactly one parse-retry"
        retry_text = bodies[1]["messages"][1]["content"][0]["text"]
        assert retry_text.endswith("Return ONLY the JSON object.")

    def test_parse_retry_failure_raises(self, key_csv, tmp_path, wav_file,
                                        presence_axis):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat_response("no json here"))

        judge = make_judge(key_csv, tmp_path, handler)
        with pytest.raises(ValueError, match="not valid JSON"):
            judge.judge(wav_file, presence_axis)

    def test_non_json_response_body_raises_and_is_journaled(self, key_csv, tmp_path,
                                                            wav_file, presence_axis):
        """A 200 with a non-JSON BODY (not assistant content) is a charged call:
        it must raise RuntimeError and still appear in the journal."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway error</html>")

        judge = make_judge(key_csv, tmp_path, handler)
        with pytest.raises(RuntimeError, match="not JSON"):
            judge.judge(wav_file, presence_axis)
        lines = (tmp_path / "mllm_cache" / "calls.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(lines) == 1


# --------------------------------------------------------------------------------------
# Budget and journal
# --------------------------------------------------------------------------------------

class TestBudget:
    def test_budget_exceeded_raises(self, key_csv, tmp_path, presence_axis):
        requests: list = []
        judge = make_judge(key_csv, tmp_path, _capture_handler(requests),
                           budget_max_calls=1)
        judge.judge(_write_wav(tmp_path / "a.wav", seed=1), presence_axis)
        with pytest.raises(MLLMBudgetExceeded):
            judge.judge(_write_wav(tmp_path / "b.wav", seed=2), presence_axis)
        assert len(requests) == 1, "the over-budget call must never reach the network"

        budget = json.loads(
            (tmp_path / "mllm_cache" / "budget.json").read_text(encoding="utf-8")
        )
        assert budget["calls"] == 1

    def test_cache_hit_consumes_no_budget(self, key_csv, tmp_path, wav_file,
                                          presence_axis):
        judge = make_judge(key_csv, tmp_path, _capture_handler([]),
                           budget_max_calls=1)
        judge.judge(wav_file, presence_axis)
        # Same wav/axis -> cache hit -> no budget charge, no raise.
        assert judge.judge(wav_file, presence_axis).label == "present"

    def test_budget_persists_across_instances(self, key_csv, tmp_path, presence_axis):
        cache_dir = tmp_path / "mllm_cache"
        make_judge(key_csv, tmp_path, _capture_handler([]), cache_dir=cache_dir,
                   budget_max_calls=1).judge(
            _write_wav(tmp_path / "a.wav", seed=1), presence_axis)
        fresh = make_judge(key_csv, tmp_path, _capture_handler([]),
                           cache_dir=cache_dir, budget_max_calls=1)
        with pytest.raises(MLLMBudgetExceeded):
            fresh.judge(_write_wav(tmp_path / "b.wav", seed=2), presence_axis)

    def test_journal_one_line_per_real_call(self, key_csv, tmp_path, wav_file,
                                            presence_axis):
        judge = make_judge(key_csv, tmp_path, _capture_handler([]))
        judge.judge(wav_file, presence_axis)
        judge.judge(wav_file, presence_axis)  # cache hit -> no new line
        lines = (tmp_path / "mllm_cache" / "calls.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["axis_id"] == "presence"
        assert entry["status"] == 200
        assert entry["attempts"] == 1


# --------------------------------------------------------------------------------------
# Per-axis schemas
# --------------------------------------------------------------------------------------

class TestSchemas:
    def test_markdown_fences_stripped(self, key_csv, tmp_path, wav_file,
                                      presence_axis):
        judge = make_judge(
            key_csv, tmp_path,
            _capture_handler([], content='```json\n{"present": false}\n```'),
        )
        assert judge.judge(wav_file, presence_axis).label == "absent"

    def test_class_axis_fills_classes_and_validates(self, key_csv, tmp_path,
                                                    wav_file, class_axis):
        requests: list = []
        judge = make_judge(
            key_csv, tmp_path,
            _capture_handler(requests, content='{"coarse_class": "impact"}'),
            coarse_classes=["impact", "scrape", "liquid"],
        )
        target = judge.judge(wav_file, class_axis)
        assert target.label == "impact"
        system_text = requests[0][1]["messages"][0]["content"]
        assert "{CLASSES}" not in system_text
        assert "impact, scrape, liquid" in system_text

    def test_class_axis_invalid_label_raises(self, key_csv, tmp_path, wav_file,
                                             class_axis):
        judge = make_judge(
            key_csv, tmp_path,
            _capture_handler([], content='{"coarse_class": "thunderstorm"}'),
            coarse_classes=["impact", "scrape"],
        )
        with pytest.raises(ValueError, match="thunderstorm"):
            judge.judge(wav_file, class_axis)

    def test_class_axis_without_class_list_requires_classes_for_template(
            self, key_csv, tmp_path, wav_file, class_axis):
        """The shipped class prompt contains {CLASSES}: judging the class axis
        without coarse_classes must fail loudly, not silently send the template."""
        judge = make_judge(key_csv, tmp_path, _capture_handler([]))
        with pytest.raises(ValueError, match="coarse_classes"):
            judge.judge(wav_file, class_axis)

    def test_timing_axis_bins_onset(self, key_csv, tmp_path, wav_file, timing_axis):
        judge = make_judge(
            key_csv, tmp_path,
            _capture_handler([], content='{"onset_s": 1.3}'),
            timing_bin_s=0.5,
        )
        target = judge.judge(wav_file, timing_axis)
        assert target.label == 2  # floor(1.3 / 0.5)

    def test_timing_axis_zero_onset(self, key_csv, tmp_path, wav_file, timing_axis):
        judge = make_judge(
            key_csv, tmp_path, _capture_handler([], content='{"onset_s": 0}'),
        )
        assert judge.judge(wav_file, timing_axis).label == 0

    def test_wrong_schema_type_raises_without_parse_retry(self, key_csv, tmp_path,
                                                          wav_file, presence_axis):
        requests: list = []
        judge = make_judge(
            key_csv, tmp_path,
            _capture_handler(requests, content='{"present": "yes"}'),
        )
        with pytest.raises(ValueError, match="boolean"):
            judge.judge(wav_file, presence_axis)
        assert len(requests) == 1, "schema violations are not parse-retried"

    def test_missing_prompt_file_raises(self, key_csv, tmp_path, wav_file):
        binding_axis = Axis(
            id="binding",
            name="multi-event binding",
            tier=AxisTier.TIER3,
            kind=AxisKind.CATEGORICAL,
            agreement=AgreementMetric.EXACT_MATCH,
            measure="binding_label",
        )
        judge = make_judge(key_csv, tmp_path, _capture_handler([]))
        with pytest.raises(FileNotFoundError, match="binding__v1"):
            judge.judge(wav_file, binding_axis)


# --------------------------------------------------------------------------------------
# test_retest
# --------------------------------------------------------------------------------------

class TestRetest:
    def test_retest_bypasses_primary_cache(self, key_csv, tmp_path, presence_axis):
        requests: list = []
        judge = make_judge(key_csv, tmp_path, _capture_handler(requests))
        wavs = [_write_wav(tmp_path / f"r{i}.wav", seed=10 + i) for i in range(2)]

        # Prime the PRIMARY cache; retest must not reuse these entries.
        for w in wavs:
            judge.judge(w, presence_axis)
        assert len(requests) == 2

        score = judge.test_retest(wavs, presence_axis, n=2)
        assert score == pytest.approx(1.0)  # identical canned response every time
        assert len(requests) == 2 + 2 * 2, "each retest copy is a real call"

        # Retest copies are themselves cached under their suffixed keys.
        score2 = judge.test_retest(wavs, presence_axis, n=2)
        assert score2 == pytest.approx(1.0)
        assert len(requests) == 6, "re-running the retest must be cache-served"

    def test_retest_detects_label_flips(self, key_csv, tmp_path, presence_axis):
        count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            count["n"] += 1
            content = '{"present": true}' if count["n"] % 2 else '{"present": false}'
            return httpx.Response(200, json=_chat_response(content))

        judge = make_judge(key_csv, tmp_path, handler)
        wav = _write_wav(tmp_path / "flip.wav", seed=3)
        score = judge.test_retest([wav], presence_axis, n=2)
        assert score == pytest.approx(0.0), "alternating labels -> zero agreement"

    def test_retest_empty_list_is_nan(self, key_csv, tmp_path, presence_axis):
        judge = make_judge(key_csv, tmp_path, _capture_handler([]))
        assert np.isnan(judge.test_retest([], presence_axis, n=2))


# --------------------------------------------------------------------------------------
# Shipped prompt files
# --------------------------------------------------------------------------------------

class TestPromptFiles:
    @pytest.mark.parametrize("axis_id", ["presence", "class", "timing"])
    def test_v1_prompt_exists_and_demands_json_only(self, axis_id):
        path = PROMPT_DIR / f"{axis_id}__v1.txt"
        assert path.exists(), f"missing prompt file {path}"
        text = path.read_text(encoding="utf-8")
        assert "ONLY the JSON object" in text

    def test_class_prompt_has_classes_placeholder(self):
        text = (PROMPT_DIR / "class__v1.txt").read_text(encoding="utf-8")
        assert "{CLASSES}" in text
