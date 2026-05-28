"""Tests for the cluster-head promotion orchestrator.

Coverage focuses on the boss-locked contracts:

* gate-arg order (champion=incumbent FIRST, challenger=candidate
  SECOND) — directional ACCEPT/REJECT regression tests catch any
  future swap.
* same-scorer / same-holdout / same-run comparability (no two
  different scorers, no two different holdouts).
* staging isolation — REJECT path leaves the pointer store
  untouched; pre-flight failures clean up the tempdir; ACCEPT path
  moves bytes into the store + flips ACTIVE.
* sidecar v1 schema (consumed by the 2d
  :class:`ClusterHeadSelector`).

Treelite is intentionally NOT loaded by these tests; ``verify_load_fn``
is injected so the suite runs in environments without libomp.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from slancha_local.train.dispatcher import DispatchError, DispatchResult
from slancha_local.train.gate import GateThresholds
from slancha_local.train.head_retrain import HeadRetrainResult
from slancha_local.train.pointer_store import PointerStore
from slancha_local.train.promote_head import (
    COMPONENT,
    HEAD_FILENAME,
    KNOWN_CAPS,
    LABEL_TABLE_FILENAME,
    SIDECAR_FILENAME,
    HeadRouter,
    HoldoutPrompt,
    PromoteHeadError,
    _build_sidecar,
    collapse_route_to_cap,
    commit_staged,
    discard_staged,
    promote_head,
    run_eval_pair,
    stage_candidate,
)
from slancha_local.train.scorer import ScoreError, ScoreResult

# -------- fakes --------


@dataclass
class _FakeDispatcher:
    """Returns a canned response for every prompt; records every call."""

    response: str = "ok"
    calls: list[tuple[str, str]] | None = None
    raise_for_model: str | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def dispatch(self, prompt: str, served_model: str) -> DispatchResult:
        self.calls.append((prompt, served_model))
        if self.raise_for_model == served_model:
            raise DispatchError(f"forced failure for {served_model}")
        return DispatchResult(
            response_text=self.response,
            served_model=served_model,
            elapsed_ms=1.0,
        )


@dataclass
class _FakeScorer:
    """Returns a deterministic score keyed off the served_model in the response.

    Tests build dispatchers that put the served_model into the
    response_text so the scorer can route a per-model score.
    Conveniently exposes ``judge_model`` so the row records a stable
    judge name across both incumbent + candidate.
    """

    judge_model: str = "judge-v1"
    score_by_model: dict[str, float] | None = None
    default_score: float = 3.0
    raise_for_response: str | None = None

    def __post_init__(self) -> None:
        if self.score_by_model is None:
            self.score_by_model = {}

    def score(self, prompt: str, response: str) -> ScoreResult:
        if self.raise_for_response == response:
            raise ScoreError(f"forced scorer failure on {response!r}")
        # response_text from _FakeDispatcher carries the served_model
        # back to the scorer in the format "served:<name>".
        if response.startswith("served:"):
            model = response.split(":", 1)[1]
        else:
            model = response
        return ScoreResult(
            score=self.score_by_model.get(model, self.default_score),
            judge_model=self.judge_model,
        )


def _make_dispatcher_echo() -> _FakeDispatcher:
    """Dispatcher whose response_text echoes the served_model.

    Combined with _FakeScorer's response-parsing, this lets a single
    fake pair score samples per-served-model deterministically.
    """
    d = _FakeDispatcher()

    def _dispatch(prompt: str, served_model: str) -> DispatchResult:
        d.calls.append((prompt, served_model))
        if d.raise_for_model == served_model:
            raise DispatchError(f"forced failure for {served_model}")
        return DispatchResult(
            response_text=f"served:{served_model}",
            served_model=served_model,
            elapsed_ms=1.0,
        )

    d.dispatch = _dispatch  # type: ignore[method-assign]
    return d


def _make_router(mapping: dict[str, str], default: str) -> HeadRouter:
    return HeadRouter(pick=lambda p: mapping.get(p, default))


def _make_holdout(n: int = 110) -> list[HoldoutPrompt]:
    """Build a min_n_eval-satisfying holdout split across two domains."""
    out: list[HoldoutPrompt] = []
    for i in range(n):
        dom = "coding" if i % 2 == 0 else "general"
        out.append(HoldoutPrompt(prompt=f"prompt-{i}", domain=dom))
    return out


def _make_head_result(label_table: list[dict] | None = None) -> HeadRetrainResult:
    if label_table is None:
        label_table = [
            {"label": 0, "route": "coding", "cluster_id": 1},
            {"label": 1, "route": "math", "cluster_id": 2},
            {"label": 2, "route": "general", "cluster_id": 3},
        ]
    return HeadRetrainResult(
        head_bytes=b"\x00fake-head-bytes",
        label_table=label_table,
        n_classes=len(label_table),
        n_samples=500,
        embedding_dim=768,
    )


def _noop_verify_load(_b: bytes) -> None:  # treelite-free
    return None


def _bad_verify_load(_b: bytes) -> None:
    raise RuntimeError("simulated treelite corruption")


# -------- sidecar shape --------


class TestBuildSidecar:
    def test_v1_schema_shape(self) -> None:
        s = _build_sidecar(
            [
                {"label": 0, "route": "coding", "cluster_id": 1},
                {"label": 1, "route": "math", "cluster_id": 7},
            ]
        )
        assert s == {
            "schema_version": "v1",
            "routes": {"1": "coding", "7": "math"},
        }

    def test_string_cluster_id_coerced_to_int_then_str(self) -> None:
        s = _build_sidecar([{"label": 0, "route": "coding", "cluster_id": "5"}])
        assert s["routes"] == {"5": "coding"}

    def test_duplicate_cluster_id_same_cap_ok(self) -> None:
        s = _build_sidecar(
            [
                {"label": 0, "route": "coding", "cluster_id": 1},
                {"label": 1, "route": "coding", "cluster_id": 1},  # idempotent dup
            ]
        )
        assert s["routes"] == {"1": "coding"}

    def test_two_raw_routes_collapsing_to_same_cap_ok(self) -> None:
        # "code" + "coding" both collapse to "coding" — not a conflict.
        s = _build_sidecar(
            [
                {"label": 0, "route": "code", "cluster_id": 1},
                {"label": 1, "route": "coding", "cluster_id": 1},
            ]
        )
        assert s["routes"] == {"1": "coding"}

    def test_duplicate_cluster_id_conflicting_cap_raises(self) -> None:
        with pytest.raises(PromoteHeadError, match="conflicting caps"):
            _build_sidecar(
                [
                    {"label": 0, "route": "coding", "cluster_id": 1},
                    {"label": 1, "route": "math", "cluster_id": 1},  # CONFLICT
                ]
            )

    def test_missing_cluster_id_raises(self) -> None:
        with pytest.raises(PromoteHeadError, match="label_table row malformed"):
            _build_sidecar([{"label": 0, "route": "coding"}])

    def test_missing_route_raises(self) -> None:
        with pytest.raises(PromoteHeadError, match="label_table row malformed"):
            _build_sidecar([{"label": 0, "cluster_id": 1}])

    # --- THE cross-phase contract test (boss's locked v1 requirement) ---

    def test_all_cap_values_are_in_known_caps_subset(self) -> None:
        """The reader (cluster_head._CLUSTER_CAP_TO_MODEL_CAP) accepts EXACTLY
        {"coding", "math", "general"}. The WRITER must defend this contract
        so the loop doesn't silently no-op on out-of-vocab clusters.

        Feed the writer the realistic upstream route formats — the
        ``classifier.route`` compound ``"<domain>_<difficulty>"`` tokens
        emitted by LocalClassifier, plus raw cap forms and the
        cluster.py "unknown" fallback; every cap value in the sidecar
        MUST be in KNOWN_CAPS or the loop dies silently for that
        cluster.
        """
        upstream_routes = [
            # LocalClassifier compound form: "<domain>_<difficulty>"
            "code_easy", "code_medium", "code_hard",
            "math_easy", "math_medium", "math_hard",
            "general_easy", "general_medium", "general_hard",
            "reasoning_hard", "creative_easy", "multilingual_medium",
            "tool-use_easy",
            # Raw cap form (defensive — if upstream ever changes)
            "coding", "math", "general",
            # cluster.py fallback when trace has no classifier output
            "unknown",
            # Adversarial: empty + arbitrary
            "", "anything-else",
        ]
        rows = [
            {"label": i, "route": r, "cluster_id": i + 1}
            for i, r in enumerate(upstream_routes)
        ]
        s = _build_sidecar(rows)
        emitted_caps = set(s["routes"].values())
        out_of_vocab = emitted_caps - KNOWN_CAPS
        assert not out_of_vocab, (
            f"sidecar emitted caps {out_of_vocab!r} outside KNOWN_CAPS — "
            f"the 2d reader will drop these and the loop will silently "
            f"no-op on those clusters"
        )
        assert emitted_caps <= KNOWN_CAPS

    def test_collapse_localclassifier_compound_routes(self) -> None:
        # LocalClassifier emits "<domain>_<difficulty>" — collapse must
        # parse the leading domain token, not match the whole string.
        # code_X → coding, math_X → math, general_X → general,
        # everything else → general.
        assert collapse_route_to_cap("code_easy") == "coding"
        assert collapse_route_to_cap("code_medium") == "coding"
        assert collapse_route_to_cap("code_hard") == "coding"
        assert collapse_route_to_cap("math_easy") == "math"
        assert collapse_route_to_cap("math_medium") == "math"
        assert collapse_route_to_cap("math_hard") == "math"
        assert collapse_route_to_cap("general_easy") == "general"
        assert collapse_route_to_cap("general_medium") == "general"
        assert collapse_route_to_cap("general_hard") == "general"
        # 7-domain non-mapped prefixes
        assert collapse_route_to_cap("reasoning_hard") == "general"
        assert collapse_route_to_cap("creative_easy") == "general"
        assert collapse_route_to_cap("multilingual_medium") == "general"
        assert collapse_route_to_cap("tool-use_easy") == "general"

    def test_collapse_route_to_cap_uses_general_for_holdout_extras(self) -> None:
        # Direct raw-domain inputs (defensive — if upstream ever changes
        # to emit bare domain tokens without difficulty suffix).
        assert collapse_route_to_cap("code") == "coding"
        assert collapse_route_to_cap("coding") == "coding"
        assert collapse_route_to_cap("math") == "math"
        for dom in ("reasoning", "creative", "multilingual", "tool-use",
                    "general", "unknown", "", "anything-else"):
            assert collapse_route_to_cap(dom) == "general", dom

    def test_collapse_is_case_insensitive_on_head(self) -> None:
        # Defensive: upstream is lowercase today but if it ever varies,
        # the collapse normalizes case so we don't accidentally fall
        # through to "general" on "Code_easy" / "MATH_hard".
        assert collapse_route_to_cap("Code_easy") == "coding"
        assert collapse_route_to_cap("MATH_hard") == "math"

    def test_out_of_vocab_collapse_raises_fail_loud(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Defense-in-depth: even if the collapse map ever drifts and
        emits a cap outside KNOWN_CAPS, _build_sidecar MUST raise at
        promote-time — not let the freshly promoted head go silently
        inert at serve-time.

        Patch the collapse table to inject a bogus mapping (simulating
        future drift) and confirm the writer-side assertion fires.
        """
        import slancha_local.train.promote_head as ph

        monkeypatch.setitem(ph._ROUTE_TO_CAP, "weird", "exotic-cap")
        with pytest.raises(PromoteHeadError, match="out-of-vocab cap"):
            _build_sidecar([{"label": 0, "route": "weird", "cluster_id": 1}])

    def test_out_of_vocab_error_message_cites_reader_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The error message must point operators at the reader-side
        # contract so they know to update both halves in lockstep.
        import slancha_local.train.promote_head as ph

        monkeypatch.setitem(ph._ROUTE_TO_CAP, "weird", "exotic-cap")
        with pytest.raises(PromoteHeadError) as exc_info:
            _build_sidecar([{"label": 0, "route": "weird", "cluster_id": 1}])
        msg = str(exc_info.value)
        assert "_CLUSTER_CAP_TO_MODEL_CAP" in msg
        assert "coding" in msg and "math" in msg and "general" in msg


# -------- staging --------


class TestStageCandidate:
    def test_writes_all_three_artifacts(self, tmp_path: Path) -> None:
        head = _make_head_result()
        staging = stage_candidate(
            head, staging_root=tmp_path, verify_load_fn=_noop_verify_load
        )
        try:
            assert (staging / HEAD_FILENAME).read_bytes() == head.head_bytes
            label_table = json.loads(
                (staging / LABEL_TABLE_FILENAME).read_text(encoding="utf-8")
            )
            assert label_table == head.label_table
            sidecar = json.loads(
                (staging / SIDECAR_FILENAME).read_text(encoding="utf-8")
            )
            assert sidecar["schema_version"] == "v1"
            assert set(sidecar["routes"].keys()) == {"1", "2", "3"}
        finally:
            discard_staged(staging)

    def test_staging_dir_isolated_from_store(self, tmp_path: Path) -> None:
        store = PointerStore(root=tmp_path / "store")
        head = _make_head_result()
        staging = stage_candidate(
            head, staging_root=tmp_path, verify_load_fn=_noop_verify_load
        )
        try:
            # Staging must NOT have polluted the pointer store.
            assert not (tmp_path / "store").exists() or not any(
                (tmp_path / "store").iterdir()
            )
            assert store.active_version(COMPONENT) is None
        finally:
            discard_staged(staging)

    def test_verify_load_failure_cleans_up_staging(self, tmp_path: Path) -> None:
        head = _make_head_result()
        before = set((tmp_path).iterdir()) if tmp_path.exists() else set()
        with pytest.raises(PromoteHeadError, match="verify-load failed"):
            stage_candidate(
                head, staging_root=tmp_path, verify_load_fn=_bad_verify_load
            )
        # No leftover promote-head-* tempdir from the failed stage.
        after = set(tmp_path.iterdir())
        assert after == before

    def test_malformed_label_table_cleans_up_staging(self, tmp_path: Path) -> None:
        head = _make_head_result(label_table=[{"label": 0, "route": "coding"}])
        before = set(tmp_path.iterdir())
        with pytest.raises(PromoteHeadError, match="label_table row malformed"):
            stage_candidate(
                head, staging_root=tmp_path, verify_load_fn=_noop_verify_load
            )
        after = set(tmp_path.iterdir())
        assert after == before


class TestCommitStaged:
    def test_writes_into_store_and_flips_active(self, tmp_path: Path) -> None:
        store = PointerStore(root=tmp_path / "store")
        head = _make_head_result()
        staging = stage_candidate(
            head, staging_root=tmp_path, verify_load_fn=_noop_verify_load
        )
        commit_staged(store, component=COMPONENT, version="20251201T000000Z", staging_dir=staging)
        assert store.active_version(COMPONENT) == "20251201T000000Z"
        active_head = store.active_path(COMPONENT, HEAD_FILENAME)
        assert active_head is not None
        assert active_head.read_bytes() == head.head_bytes
        active_side = store.active_path(COMPONENT, SIDECAR_FILENAME)
        assert active_side is not None
        # Staging dir is rmtree'd after commit.
        assert not staging.exists()

    def test_empty_staging_raises(self, tmp_path: Path) -> None:
        store = PointerStore(root=tmp_path / "store")
        empty = tmp_path / "empty-staging"
        empty.mkdir()
        with pytest.raises(PromoteHeadError, match="nothing to commit"):
            commit_staged(
                store, component=COMPONENT, version="20251201T000000Z", staging_dir=empty
            )


# -------- eval pair --------


class TestRunEvalPair:
    def test_uses_both_routers_for_every_prompt(self) -> None:
        # Incumbent picks model A, candidate picks model B.
        inc = _make_router({}, default="modelA")
        cand = _make_router({}, default="modelB")
        d = _make_dispatcher_echo()
        s = _FakeScorer(score_by_model={"modelA": 3.0, "modelB": 4.0})

        holdout = _make_holdout(n=110)
        inc_row, cand_row = run_eval_pair(
            holdout,
            incumbent_router=inc,
            candidate_router=cand,
            dispatcher=d,
            scorer=s,
            incumbent_version="inc-v1",
            candidate_version="cand-v1",
            holdout_version=1,
        )
        # Each prompt → exactly 2 dispatch calls (one per router).
        assert len(d.calls) == 220
        # Both rows aggregate to the per-model means we wired.
        assert inc_row["mean_score"] == 3.0
        assert cand_row["mean_score"] == 4.0
        assert inc_row["judge_model"] == cand_row["judge_model"] == "judge-v1"
        assert inc_row["holdout_version"] == cand_row["holdout_version"] == 1
        assert inc_row["router_version"] == "inc-v1"
        assert cand_row["router_version"] == "cand-v1"

    def test_dispatch_failure_counted_not_aborts(self) -> None:
        inc = _make_router({}, default="modelA")
        cand = _make_router({}, default="badmodel")
        d = _make_dispatcher_echo()
        d.raise_for_model = "badmodel"
        s = _FakeScorer(score_by_model={"modelA": 4.0})

        holdout = _make_holdout(n=110)
        inc_row, cand_row = run_eval_pair(
            holdout,
            incumbent_router=inc,
            candidate_router=cand,
            dispatcher=d,
            scorer=s,
            incumbent_version="inc",
            candidate_version="cand",
            holdout_version=1,
        )
        assert inc_row["n_dispatch_failures"] == 0
        assert cand_row["n_dispatch_failures"] == 110  # every candidate call failed

    def test_scorer_failure_counted_not_aborts(self) -> None:
        inc = _make_router({}, default="modelA")
        cand = _make_router({}, default="modelB")
        d = _make_dispatcher_echo()
        s = _FakeScorer(score_by_model={"modelA": 4.0, "modelB": 4.0})
        s.raise_for_response = "served:modelB"  # every candidate response trips scorer

        holdout = _make_holdout(n=110)
        inc_row, cand_row = run_eval_pair(
            holdout,
            incumbent_router=inc,
            candidate_router=cand,
            dispatcher=d,
            scorer=s,
            incumbent_version="inc",
            candidate_version="cand",
            holdout_version=1,
        )
        assert inc_row["n_scorer_failures"] == 0
        assert cand_row["n_scorer_failures"] == 110

    def test_same_scorer_means_same_judge_model_on_both_rows(self) -> None:
        # The "comparability by construction" guarantee.
        inc = _make_router({}, default="modelA")
        cand = _make_router({}, default="modelB")
        d = _make_dispatcher_echo()
        s = _FakeScorer(judge_model="judge-pinned-X")
        inc_row, cand_row = run_eval_pair(
            _make_holdout(n=110),
            incumbent_router=inc,
            candidate_router=cand,
            dispatcher=d,
            scorer=s,
            incumbent_version="inc",
            candidate_version="cand",
            holdout_version=1,
        )
        assert inc_row["judge_model"] == "judge-pinned-X"
        assert cand_row["judge_model"] == "judge-pinned-X"


# -------- promote_head end-to-end --------


def _setup_pipeline(
    *,
    tmp_path: Path,
    inc_score: float,
    cand_score: float,
    promotions_log: Path | None = None,
    dry_run: bool = False,
    label_table: list[dict] | None = None,
):
    store = PointerStore(root=tmp_path / "store")
    head = _make_head_result(label_table=label_table)
    inc_router = _make_router({}, default="modelA")
    cand_router_seen_paths: list[Path] = []

    def factory(staging_dir: Path) -> HeadRouter:
        cand_router_seen_paths.append(staging_dir)
        return _make_router({}, default="modelB")

    d = _make_dispatcher_echo()
    s = _FakeScorer(score_by_model={"modelA": inc_score, "modelB": cand_score})
    verdict = promote_head(
        store,
        head_result=head,
        holdout=_make_holdout(n=110),
        incumbent_router=inc_router,
        candidate_router_factory=factory,
        dispatcher=d,
        scorer=s,
        holdout_version=1,
        thresholds=GateThresholds(mean_score_delta=0.05, min_n_eval=100),
        promotions_log=promotions_log,
        dry_run=dry_run,
        staging_root=tmp_path,
        verify_load_fn=_noop_verify_load,
    )
    return store, verdict, cand_router_seen_paths


class TestPromoteHeadDirectional:
    """The gate-arg-order guard the boss specifically requested.

    If anyone ever swaps champion/challenger in promote_head's
    decide() call, mean_delta inverts and these two tests flip. They
    are the ONE thing that must keep passing forever.
    """

    def test_clearly_better_candidate_must_accept(self, tmp_path: Path) -> None:
        store, verdict, _ = _setup_pipeline(
            tmp_path=tmp_path, inc_score=3.0, cand_score=4.5
        )
        assert verdict.accept, f"better candidate must ACCEPT; got reasons={verdict.reject_reasons}"
        assert verdict.mean_delta == pytest.approx(1.5)
        # And the store flipped to the candidate.
        assert store.active_version(COMPONENT) is not None
        assert store.active_path(COMPONENT, HEAD_FILENAME) is not None

    def test_clearly_worse_candidate_must_reject(self, tmp_path: Path) -> None:
        store, verdict, _ = _setup_pipeline(
            tmp_path=tmp_path, inc_score=4.5, cand_score=3.0
        )
        assert not verdict.accept, "worse candidate must REJECT — if not, args are swapped"
        assert verdict.mean_delta == pytest.approx(-1.5)
        # ACTIVE must be untouched on REJECT.
        assert store.active_version(COMPONENT) is None


class TestPromoteHeadAcceptPath:
    def test_accept_commits_head_and_sidecar(self, tmp_path: Path) -> None:
        store, verdict, _ = _setup_pipeline(
            tmp_path=tmp_path, inc_score=3.0, cand_score=4.5
        )
        assert verdict.accept
        head_path = store.active_path(COMPONENT, HEAD_FILENAME)
        side_path = store.active_path(COMPONENT, SIDECAR_FILENAME)
        assert head_path is not None and head_path.exists()
        assert side_path is not None and side_path.exists()
        # Sidecar matches the v1 schema.
        side = json.loads(side_path.read_text(encoding="utf-8"))
        assert side["schema_version"] == "v1"
        assert side["routes"] == {"1": "coding", "2": "math", "3": "general"}
        # Head + sidecar live in the SAME version dir.
        assert head_path.parent == side_path.parent

    def test_accept_appends_verdict_when_log_set(self, tmp_path: Path) -> None:
        log = tmp_path / "promotions.jsonl"
        _, verdict, _ = _setup_pipeline(
            tmp_path=tmp_path, inc_score=3.0, cand_score=4.5, promotions_log=log
        )
        assert verdict.accept
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["accept"] is True


class TestPromoteHeadRejectPath:
    def test_reject_leaves_pointer_store_untouched(self, tmp_path: Path) -> None:
        store, verdict, _ = _setup_pipeline(
            tmp_path=tmp_path, inc_score=4.5, cand_score=3.0
        )
        assert not verdict.accept
        # No version dirs created.
        cdir = store.component_dir(COMPONENT)
        if cdir.exists():
            assert not list(cdir.iterdir())
        assert store.active_version(COMPONENT) is None

    def test_reject_appends_verdict_when_log_set(self, tmp_path: Path) -> None:
        log = tmp_path / "promotions.jsonl"
        _, verdict, _ = _setup_pipeline(
            tmp_path=tmp_path, inc_score=4.5, cand_score=3.0, promotions_log=log
        )
        assert not verdict.accept
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["accept"] is False

    def test_reject_discards_staging_dir(self, tmp_path: Path) -> None:
        _, _, seen_paths = _setup_pipeline(
            tmp_path=tmp_path, inc_score=4.5, cand_score=3.0
        )
        # The candidate_router_factory saw the staging dir, but it
        # must be gone after promote_head returns.
        for p in seen_paths:
            assert not p.exists(), f"REJECT must rmtree staging dir {p}"


class TestPromoteHeadDryRun:
    def test_dry_run_accept_does_not_write_to_store(self, tmp_path: Path) -> None:
        store, verdict, _ = _setup_pipeline(
            tmp_path=tmp_path, inc_score=3.0, cand_score=4.5, dry_run=True
        )
        assert verdict.accept  # gate still computes correctly
        assert store.active_version(COMPONENT) is None  # but nothing committed
        cdir = store.component_dir(COMPONENT)
        if cdir.exists():
            assert not list(cdir.iterdir())

    def test_dry_run_never_appends_to_promotions_log(self, tmp_path: Path) -> None:
        log = tmp_path / "promotions.jsonl"
        _, _, _ = _setup_pipeline(
            tmp_path=tmp_path,
            inc_score=3.0,
            cand_score=4.5,
            promotions_log=log,
            dry_run=True,
        )
        assert not log.exists(), "dry-run must NOT write the promotions log"


class TestPromoteHeadPreFlight:
    def test_verify_load_failure_cleans_up_and_raises(self, tmp_path: Path) -> None:
        store = PointerStore(root=tmp_path / "store")
        head = _make_head_result()
        with pytest.raises(PromoteHeadError, match="verify-load failed"):
            promote_head(
                store,
                head_result=head,
                holdout=_make_holdout(n=110),
                incumbent_router=_make_router({}, default="modelA"),
                candidate_router_factory=lambda _p: _make_router({}, default="modelB"),
                dispatcher=_make_dispatcher_echo(),
                scorer=_FakeScorer(),
                holdout_version=1,
                staging_root=tmp_path,
                verify_load_fn=_bad_verify_load,
            )
        assert store.active_version(COMPONENT) is None

    def test_incumbent_version_falls_back_to_active(self, tmp_path: Path) -> None:
        # Seed the store with an existing ACTIVE version.
        store = PointerStore(root=tmp_path / "store")
        store.write_candidate(COMPONENT, "20250101T000000Z", {"old.bin": b"old"})
        store.promote(COMPONENT, "20250101T000000Z")

        head = _make_head_result()
        verdict = promote_head(
            store,
            head_result=head,
            holdout=_make_holdout(n=110),
            incumbent_router=_make_router({}, default="modelA"),
            candidate_router_factory=lambda _p: _make_router({}, default="modelB"),
            dispatcher=_make_dispatcher_echo(),
            scorer=_FakeScorer(score_by_model={"modelA": 3.0, "modelB": 4.5}),
            holdout_version=1,
            staging_root=tmp_path,
            verify_load_fn=_noop_verify_load,
        )
        assert verdict.champion_version == "20250101T000000Z"

    def test_candidate_router_factory_called_with_staging_dir(
        self, tmp_path: Path
    ) -> None:
        _, _, seen_paths = _setup_pipeline(
            tmp_path=tmp_path, inc_score=3.0, cand_score=4.5
        )
        assert len(seen_paths) == 1
        # Factory was called with the staging dir BEFORE eval; that
        # dir is gone now (committed-moved-and-rmtree'd), but the
        # callback got it at the right time.

    def test_thresholds_default_constructs_when_omitted(self, tmp_path: Path) -> None:
        store = PointerStore(root=tmp_path / "store")
        head = _make_head_result()
        verdict = promote_head(
            store,
            head_result=head,
            holdout=_make_holdout(n=110),
            incumbent_router=_make_router({}, default="modelA"),
            candidate_router_factory=lambda _p: _make_router({}, default="modelB"),
            dispatcher=_make_dispatcher_echo(),
            scorer=_FakeScorer(score_by_model={"modelA": 3.0, "modelB": 3.06}),
            holdout_version=1,
            staging_root=tmp_path,
            verify_load_fn=_noop_verify_load,
        )
        # Default mean_score_delta = 0.05; 0.06 lift > 0.05, so ACCEPT.
        assert verdict.accept
