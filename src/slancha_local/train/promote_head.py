"""Phase 2c: cluster-head promotion orchestrator (staging-dir flow).

Closes the cluster-head training loop. Takes a freshly retrained
``HeadRetrainResult`` from :mod:`slancha_local.train.head_retrain`,
stages it in a temporary directory, evaluates BOTH the incumbent and
the challenger on the same holdout in the same run (so the
comparison is comparable by construction), runs
:func:`slancha_local.train.gate.decide` with the **incumbent as
champion and the challenger as candidate**, and either commits the
candidate into the :class:`PointerStore` + flips ACTIVE (ACCEPT) or
deletes the staging dir untouched (REJECT). Either way it appends a
verdict to the promotions log.

The CRITICAL CONTRACTS this module enforces:

1. **Staging is isolated.** All candidate I/O happens in a
   :func:`tempfile.mkdtemp` directory. ``write_candidate`` (which
   places files under the pointer store and could be observed by a
   concurrent reader) is called ONLY after the candidate passes
   ``verify_load``. A REJECT verdict never touches the pointer
   store — ``rmtree`` the staging dir and move on.

2. **Sidecar is co-located.** ``cluster_id_to_route.json`` (schema
   v1) is written into the SAME directory as
   ``mmbert_tl_cluster.bin`` so the pointer-store ACTIVE→version
   resolution lands both files atomically. The
   :class:`ClusterHeadSelector` from
   :mod:`slancha_local.classifier.cluster_head` is the consumer.

3. **Gate args are not swapped.** ``decide(champion=incumbent_row,
   challenger=candidate_row)`` — incumbent FIRST. A swap would
   silently invert ``mean_delta`` and start accepting WORSE heads.
   :func:`promote_head` defends this with type-level kwargs and the
   directional regression tests in
   ``tests/unit/test_train_promote_head.py``.

4. **Judge consistency by construction.** A SINGLE
   :class:`~slancha_local.train.scorer.Scorer` instance scores both
   the incumbent's responses and the candidate's responses in the
   same loop. ``gate.require_judge_match`` therefore passes
   trivially — no stale-row + cross-judge bugs possible.

5. **Holdout sha256-only.** The orchestrator does not copy holdout
   prompts anywhere; it consumes them in-memory + records the
   ``holdout_version`` on both rows. The user is responsible for
   pinning the holdout source.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slancha_local.train.dispatcher import Dispatcher, DispatchError
from slancha_local.train.eval_row import EvalSample, aggregate_eval_pass
from slancha_local.train.gate import (
    GateThresholds,
    PromotionVerdict,
    append_verdict,
    decide,
)
from slancha_local.train.head_retrain import HeadRetrainResult, verify_load
from slancha_local.train.pointer_store import PointerStore, new_version
from slancha_local.train.scorer import ScoreError, Scorer

logger = logging.getLogger(__name__)


# Constants chosen to match the 2d READ path's contracts.
HEAD_FILENAME = "mmbert_tl_cluster.bin"
SIDECAR_FILENAME = "cluster_id_to_route.json"
LABEL_TABLE_FILENAME = "label_table.json"
COMPONENT = "classifier-head"
SCHEMA_VERSION = "v1"

# The READ side
# (:mod:`slancha_local.classifier.cluster_head._CLUSTER_CAP_TO_MODEL_CAP`)
# accepts EXACTLY this cap vocabulary; any other cap → ``_apply_cluster_hint``
# logs "unknown cap" and falls through, so the cluster never influences
# routing despite a promoted head. The WRITER must defend the contract by
# collapsing every possible upstream route/domain value into one of these
# three before writing the sidecar. Expanding the vocabulary later requires
# changing BOTH the reader (cluster_head.py + local._CLUSTER_CAP_TO_MODEL_CAP)
# AND this writer in the same change.
KNOWN_CAPS: frozenset[str] = frozenset({"coding", "math", "general"})

# Collapse map for the v1 sidecar. Applied to the leading domain
# token of ``route`` (split on ``_``), because the upstream
# ``classifier.route`` is the compound ``"<domain>_<difficulty>"`` form
# emitted by :class:`~slancha_local.classifier.local.LocalClassifier`
# (e.g. ``"code_easy"``, ``"math_hard"``, ``"general_medium"``). Also
# accepts the raw cap forms (``"coding"`` / already-cap) defensively in
# case the upstream ever changes to emit caps directly. Everything not
# explicitly mapped collapses to ``"general"``.
_ROUTE_HEAD_TO_CAP: dict[str, str] = {
    "code": "coding",
    "coding": "coding",
    "math": "math",
}

# Backwards-compatible alias: ``_ROUTE_TO_CAP`` is the public-ish symbol
# tests monkeypatch to simulate vocabulary drift. Same dict, same mapping;
# accessed via the head-token lookup in :func:`collapse_route_to_cap`.
_ROUTE_TO_CAP = _ROUTE_HEAD_TO_CAP


def collapse_route_to_cap(route: str) -> str:
    """Collapse an upstream classifier ``route`` to a v1 sidecar cap.

    The upstream ``classifier.route`` (see
    :meth:`slancha_local.classifier.local.LocalClassifier.classify`) is
    the compound ``"<domain>_<difficulty>"`` token; ``cluster_by_route``
    groups traces by that string verbatim and the label_table carries
    it through. We collapse on the leading domain token:

    * ``code*``, ``coding`` → ``"coding"``
    * ``math*`` → ``"math"``
    * everything else (``general*``, ``reasoning*``, ``creative*``,
      ``multilingual*``, ``tool-use*``, ``unknown``, ...) → ``"general"``

    Returns one of :data:`KNOWN_CAPS` — never anything else. The reader
    (:class:`~slancha_local.classifier.cluster_head.ClusterHeadSelector`)
    drops any cap outside that set, so the writer is the place to
    enforce the vocabulary or the loop silently no-ops on
    out-of-vocab clusters.
    """
    head = route.split("_", 1)[0].lower()
    return _ROUTE_TO_CAP.get(head, "general")


class PromoteHeadError(RuntimeError):
    """Raised when the promotion pipeline cannot be started.

    Used for pre-flight errors (missing holdout, malformed head
    result). Mid-pipeline per-sample dispatch / scorer failures are
    counted into the eval-row instead — the orchestrator continues
    so one bad prompt doesn't doom the whole pass.
    """


@dataclass(frozen=True)
class HoldoutPrompt:
    """One holdout prompt with its domain tag.

    ``domain`` is the same string the rest of slancha-local uses for
    routing (``_DOMAIN_TO_CAP`` keys); it feeds the gate's
    ``per_domain_mean`` so per-domain regressions are catchable.
    """

    prompt: str
    domain: str


@dataclass(frozen=True)
class HeadRouter:
    """Maps a holdout prompt to the served_model that head would pick.

    Two of these are needed per promotion run: one bound to the
    incumbent ACTIVE artifacts, one bound to the candidate staging
    dir. The dispatcher then dispatches to whichever model is
    returned. Implementations may be:

    * a thin wrapper around :class:`~slancha_local.classifier.local.LocalClassifier`
      + a :class:`~slancha_local.classifier.cluster_head.ClusterHeadSelector`
      pinned to a specific head artifact, or
    * a fixture map ``{prompt -> served_model}`` for tests.

    The Protocol is intentionally tiny: ``pick(prompt) -> served_model``.
    """

    pick: Callable[[str], str]


def _build_sidecar(label_table: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the ``cluster_id_to_route.json`` v1 payload from a label_table.

    ``label_table`` rows are ``{"label": int, "route": str,
    "cluster_id": int}``. Each row's ``route`` is collapsed to a v1
    cap via :func:`collapse_route_to_cap`; the sidecar's ``routes``
    values are then **defensively asserted** to be a subset of
    :data:`KNOWN_CAPS` and a vocabulary mismatch raises
    :class:`PromoteHeadError` at promote-time so the operator sees
    the error immediately, rather than the freshly promoted head
    going SILENTLY INERT at serve-time (which is what would happen
    if the 2d reader saw a cap outside its accepted vocabulary —
    :data:`slancha_local.classifier.cluster_head._CLUSTER_CAP_TO_MODEL_CAP`
    would WARN-drop it and ``_apply_cluster_hint`` would fall
    through, no-op'ing the cluster).

    Conflict detection runs against the COLLAPSED caps, not the raw
    routes — two label_table rows mapping the same cluster_id to
    different caps is an upstream bug worth raising on; two rows
    mapping the same cluster_id to the same cap (idempotent dups,
    or two raw routes that collapse to the same cap) is fine.

    Raises :class:`PromoteHeadError` if a row is malformed, has
    conflicting caps for the same cluster_id, or — defensively —
    if any collapsed cap somehow ends up outside :data:`KNOWN_CAPS`
    (which would mean the writer/reader contract has drifted and
    needs to be re-locked in lockstep).
    """
    routes: dict[str, str] = {}
    for row in label_table:
        try:
            cid = int(row["cluster_id"])
            raw_route = str(row["route"])
        except (KeyError, TypeError, ValueError) as e:
            raise PromoteHeadError(
                f"label_table row malformed (need cluster_id + route): {row!r}"
            ) from e
        cap = collapse_route_to_cap(raw_route)
        # Defense-in-depth: collapse_route_to_cap currently always
        # returns a value in KNOWN_CAPS, but the writer/reader contract
        # is too important to leave to a single function's behavior.
        # If the collapse map ever drifts (e.g., a new entry maps to a
        # non-cap), fail loud at promote-time, not silently at serve-time.
        if cap not in KNOWN_CAPS:
            raise PromoteHeadError(
                f"label_table row produced out-of-vocab cap {cap!r} from "
                f"route {raw_route!r}; the 2d reader "
                f"(classifier.cluster_head._CLUSTER_CAP_TO_MODEL_CAP) "
                f"accepts EXACTLY {sorted(KNOWN_CAPS)!r} — expanding the "
                f"vocabulary requires updating both reader and writer "
                f"(promote_head.KNOWN_CAPS + _ROUTE_TO_CAP) in the same change"
            )
        key = str(cid)
        if key in routes and routes[key] != cap:
            raise PromoteHeadError(
                f"label_table has conflicting caps for cluster_id={cid}: "
                f"{routes[key]!r} vs {cap!r}"
            )
        routes[key] = cap
    return {"schema_version": SCHEMA_VERSION, "routes": routes}


def stage_candidate(
    head_result: HeadRetrainResult,
    *,
    staging_root: Path | None = None,
    verify_load_fn: Callable[[bytes], None] = verify_load,
) -> Path:
    """Write a candidate's artifacts to a fresh tempdir, return the dir.

    Writes three files into the staging dir:

    * ``mmbert_tl_cluster.bin`` — head bytes (treelite-serialized)
    * ``label_table.json`` — index→(route, cluster_id) decoder rows
    * ``cluster_id_to_route.json`` — sidecar consumed by the 2d
      :class:`~slancha_local.classifier.cluster_head.ClusterHeadSelector`

    Runs ``verify_load_fn`` on the head bytes BEFORE returning so
    the orchestrator catches corruption before any expensive eval
    work. ``verify_load_fn`` is injectable for tests that lack a
    treelite native library — the default is the real
    :func:`slancha_local.train.head_retrain.verify_load`.

    Raises :class:`PromoteHeadError` if verify-load fails (the
    staging dir is rmtree'd before raising so a failed stage leaves
    no garbage).
    """
    staging_dir = Path(tempfile.mkdtemp(prefix="promote-head-", dir=staging_root))
    try:
        (staging_dir / HEAD_FILENAME).write_bytes(head_result.head_bytes)
        (staging_dir / LABEL_TABLE_FILENAME).write_text(
            json.dumps(head_result.label_table, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        sidecar = _build_sidecar(head_result.label_table)
        (staging_dir / SIDECAR_FILENAME).write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            verify_load_fn(head_result.head_bytes)
        except Exception as e:  # noqa: BLE001
            raise PromoteHeadError(f"candidate verify-load failed: {e}") from e
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return staging_dir


def discard_staged(staging_dir: Path) -> None:
    """Remove ``staging_dir`` and its contents (REJECT path)."""
    shutil.rmtree(staging_dir, ignore_errors=True)


def commit_staged(
    store: PointerStore,
    *,
    component: str,
    version: str,
    staging_dir: Path,
) -> Path:
    """Move all files in ``staging_dir`` into ``store`` as version ``version``.

    Calls :meth:`PointerStore.write_candidate` with the file payloads
    read from staging, then :meth:`PointerStore.promote` to flip
    ACTIVE. The staging dir is rmtree'd on success.

    A crash between ``write_candidate`` and ``promote`` is safe: the
    candidate version dir lives on disk but ACTIVE has not flipped,
    so the loader keeps using the prior version. The next run will
    notice + either re-promote or prune via ``keep_versions``.
    """
    files: dict[str, bytes] = {}
    for entry in sorted(staging_dir.iterdir()):
        if not entry.is_file():
            continue
        files[entry.name] = entry.read_bytes()
    if not files:
        raise PromoteHeadError(f"nothing to commit from empty staging dir {staging_dir}")
    vdir = store.write_candidate(component, version, files)
    store.promote(component, version)
    shutil.rmtree(staging_dir, ignore_errors=True)
    return vdir


def run_eval_pair(
    holdout: Iterable[HoldoutPrompt],
    *,
    incumbent_router: HeadRouter,
    candidate_router: HeadRouter,
    dispatcher: Dispatcher,
    scorer: Scorer,
    incumbent_version: str,
    candidate_version: str,
    holdout_version: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score both routers on the same holdout in the same run.

    The "two rows from one run" pattern is what makes the gate's
    judge-match check pass trivially — both rows are produced by the
    SAME scorer instance, so ``judge_model`` matches by
    construction. Dispatch failures are counted into the row as
    ``failure_kind="dispatch"`` samples (score 0); scorer failures
    as ``failure_kind="scorer"`` (score 0). One bad prompt never
    aborts the pass.

    Returns ``(champion_row, challenger_row)`` ready for
    :func:`gate.decide`. ``champion_row`` = incumbent (the row that
    must come FIRST in the gate call); ``challenger_row`` =
    candidate.
    """
    incumbent_samples: list[EvalSample] = []
    candidate_samples: list[EvalSample] = []
    started = time.perf_counter()

    for hp in holdout:
        for router, samples in (
            (incumbent_router, incumbent_samples),
            (candidate_router, candidate_samples),
        ):
            served_model = router.pick(hp.prompt)
            try:
                dr = dispatcher.dispatch(hp.prompt, served_model)
            except DispatchError as e:
                logger.warning(
                    "dispatch failed for prompt domain=%s model=%s: %s",
                    hp.domain,
                    served_model,
                    e,
                )
                samples.append(
                    EvalSample(
                        domain=hp.domain,
                        served_model=served_model,
                        score=0.0,
                        failure_kind="dispatch",
                    )
                )
                continue
            try:
                sr = scorer.score(hp.prompt, dr.response_text)
            except ScoreError as e:
                logger.warning(
                    "scorer failed for prompt domain=%s model=%s: %s",
                    hp.domain,
                    served_model,
                    e,
                )
                samples.append(
                    EvalSample(
                        domain=hp.domain,
                        served_model=served_model,
                        score=0.0,
                        failure_kind="scorer",
                    )
                )
                continue
            samples.append(
                EvalSample(
                    domain=hp.domain,
                    served_model=served_model,
                    score=sr.score,
                )
            )

    elapsed_seconds = time.perf_counter() - started

    # Heuristic: the judge_model comes from the FIRST successful
    # score (any sample, either side — they're produced by the same
    # scorer instance). If every sample failed, fall back to the
    # scorer's class name so the row still records something
    # parseable for the operator.
    judge_model = _first_successful_judge(incumbent_samples, candidate_samples, scorer)

    incumbent_row = aggregate_eval_pass(
        incumbent_samples,
        router_version=incumbent_version,
        judge_model=judge_model,
        holdout_version=holdout_version,
        elapsed_seconds=elapsed_seconds,
    )
    candidate_row = aggregate_eval_pass(
        candidate_samples,
        router_version=candidate_version,
        judge_model=judge_model,
        holdout_version=holdout_version,
        elapsed_seconds=elapsed_seconds,
    )
    return incumbent_row, candidate_row


def _first_successful_judge(
    a: list[EvalSample], b: list[EvalSample], scorer: Scorer
) -> str:
    """Pull the judge model name off the scorer (if exposed) for the row.

    The :class:`Scorer` Protocol only constrains ``.score()``; we
    introspect ``judge_model`` attribute as a courtesy for the
    built-in :class:`~slancha_local.train.scorer.HttpxLocalJudgeScorer`
    + the fixture fakes used in tests. Falls back to the scorer's
    class name when no attribute exists — the row still has SOMETHING
    parseable + ``gate.require_judge_match`` still passes because
    both rows used the same scorer instance.
    """
    explicit = getattr(scorer, "judge_model", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    return type(scorer).__name__


def promote_head(
    store: PointerStore,
    *,
    head_result: HeadRetrainResult,
    holdout: Iterable[HoldoutPrompt],
    incumbent_router: HeadRouter,
    candidate_router_factory: Callable[[Path], HeadRouter],
    dispatcher: Dispatcher,
    scorer: Scorer,
    holdout_version: int,
    incumbent_version: str | None = None,
    component: str = COMPONENT,
    thresholds: GateThresholds | None = None,
    promotions_log: Path | None = None,
    dry_run: bool = False,
    staging_root: Path | None = None,
    verify_load_fn: Callable[[bytes], None] = verify_load,
    now_fn: Callable[[], str] = new_version,
) -> PromotionVerdict:
    """End-to-end: stage → verify → eval-both → gate → commit-or-discard.

    Pipeline:

    1. ``stage_candidate`` writes head + label_table + sidecar to a
       fresh tempdir and ``verify_load``s the head bytes.
    2. ``candidate_router_factory(staging_dir)`` produces the
       :class:`HeadRouter` bound to the staged artifact.
    3. ``run_eval_pair`` scores incumbent + candidate on the SAME
       holdout with the SAME scorer in the SAME run.
    4. ``gate.decide(champion=incumbent_row, challenger=candidate_row)``
       — incumbent FIRST (a swap here would silently invert
       ``mean_delta`` and start accepting WORSE heads; the
       directional tests catch this).
    5. ACCEPT path: ``commit_staged`` (write_candidate → promote →
       rmtree staging). REJECT path: ``discard_staged`` (rmtree
       staging, ACTIVE untouched). EITHER way the verdict is
       appended to ``promotions_log`` when supplied.

    ``dry_run=True`` runs the whole pipeline (including eval +
    gate) but never writes to the pointer store + never appends to
    the promotions log. Useful for ``slancha promote-head --dry-run``
    when the operator wants to see what WOULD happen.

    Raises :class:`PromoteHeadError` only for pre-flight problems
    (label_table malformed, verify-load fails, empty staging at
    commit time). All transport-level errors are absorbed into the
    eval-row's failure counters by :func:`run_eval_pair`.
    """
    if thresholds is None:
        thresholds = GateThresholds()
    if incumbent_version is None:
        incumbent_version = store.active_version(component) or "none"

    staging_dir = stage_candidate(
        head_result,
        staging_root=staging_root,
        verify_load_fn=verify_load_fn,
    )

    try:
        candidate_version = now_fn()
        candidate_router = candidate_router_factory(staging_dir)

        champion_row, challenger_row = run_eval_pair(
            holdout,
            incumbent_router=incumbent_router,
            candidate_router=candidate_router,
            dispatcher=dispatcher,
            scorer=scorer,
            incumbent_version=incumbent_version,
            candidate_version=candidate_version,
            holdout_version=holdout_version,
        )
        # GATE-ARG ORDER: champion=incumbent, challenger=candidate.
        # A swap here flips mean_delta sign + starts accepting WORSE
        # heads silently. The directional regression tests in
        # tests/unit/test_train_promote_head.py guard this contract.
        verdict = decide(
            champion=champion_row,
            challenger=challenger_row,
            thresholds=thresholds,
        )

        if verdict.accept and not dry_run:
            commit_staged(
                store,
                component=component,
                version=candidate_version,
                staging_dir=staging_dir,
            )
            logger.info(
                "promote_head: ACCEPT %s -> %s (mean_delta=%+.4f)",
                incumbent_version,
                candidate_version,
                verdict.mean_delta,
            )
        else:
            discard_staged(staging_dir)
            if verdict.accept and dry_run:
                logger.info(
                    "promote_head: ACCEPT (dry-run) %s -> %s (mean_delta=%+.4f) — "
                    "staging dir discarded, ACTIVE untouched",
                    incumbent_version,
                    candidate_version,
                    verdict.mean_delta,
                )
            else:
                logger.info(
                    "promote_head: REJECT %s -> %s (mean_delta=%+.4f, reasons=%s)",
                    incumbent_version,
                    candidate_version,
                    verdict.mean_delta,
                    list(verdict.reject_reasons),
                )
    except Exception:
        discard_staged(staging_dir)
        raise

    if promotions_log is not None and not dry_run:
        append_verdict(promotions_log, verdict)

    return verdict
