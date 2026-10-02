from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from eval import runtime_evaluator
from eval.runtime_evaluator import RuntimeEvaluator


def _failing_evaluation(**kwargs):  # noqa: ANN003, ANN202
    raise ValueError("scorer crashed")


@pytest.mark.asyncio
async def test_evaluation_failure_fails_the_rollout(monkeypatch, tmp_path) -> None:
    """A crashing scorer must fail the rollout, not silently drop its scene.

    The worker reports a failed rollout, which aggregation scores as 0; a
    swallowed error would leave the scene out of the score altogether.
    """
    eval_config = SimpleNamespace(
        enabled=True, video=SimpleNamespace(render_video=False)
    )
    evaluator = RuntimeEvaluator.__new__(RuntimeEvaluator)
    evaluator.eval_config = eval_config
    evaluator.rollout_uuid = "rollout-1"
    evaluator.scene_id = "clipgt-x"
    evaluator.save_path_root = str(tmp_path)
    monkeypatch.setattr(evaluator, "build_eval_input", lambda: object())
    monkeypatch.setattr(
        runtime_evaluator, "_evaluate_in_subprocess", _failing_evaluation
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(
            RuntimeError, match="Evaluation failed for session rollout-1"
        ):
            await evaluator.run_evaluation(executor)
