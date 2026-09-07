from io import StringIO

from rich.console import Console

from videoclean.progress import JobProgress


def _printed(job: JobProgress) -> str:
    buf = StringIO()
    Console(file=buf, width=140, force_terminal=False, no_color=True).print(job.render())
    return buf.getvalue()


def test_tick_detail_is_visible_on_running_stage():
    job = JobProgress(job_id="job1", headline="clip", estimated_seconds=60)
    job.start("verify", total=8, detail="read inpainted frames")
    job.tick("verify", 3, 8, "grounding-dino keyframe 3/8  frame 15/40")
    out = _printed(job)
    assert "Проверка" in out
    assert "grounding-dino keyframe 3/8" in out
    assert "frame 15/40" in out


def test_live_uses_render_callable_not_a_frozen_snapshot():
    job = JobProgress(job_id="job1", headline="clip", estimated_seconds=60)
    assert job._live is None
    with job:
        assert job._live is not None
        assert job._live._get_renderable == job.render
        job.start("verify", total=8, detail="re-detect leftover")
        first = _printed(job)
        job.tick("verify", 4, 8, "grounding-dino keyframe 4/8  frame 20/40")
        second = _printed(job)
    assert "re-detect leftover" in first
    assert "keyframe 4/8" in second
