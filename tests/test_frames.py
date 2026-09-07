from pathlib import Path

import numpy as np

from videoclean.application.frames import LazyFrames


def test_lazy_frames_cache_evicts(tmp_path: Path):
    reads: list[int] = []

    def read(path: Path):
        idx = int(path.stem.split("_")[-1])
        reads.append(idx)
        return np.full((2, 2, 3), idx, dtype=np.uint8)

    paths = [tmp_path / f"frame_{i:06d}.jpg" for i in range(6)]
    frames = LazyFrames(paths, read, cache_size=2)
    assert frames[0][0, 0, 0] == 0
    assert frames[1][0, 0, 0] == 1
    assert frames[2][0, 0, 0] == 2
    assert reads == [0, 1, 2]
    _ = frames[0]
    assert reads[-1] == 0
    assert len(frames._cache) <= 2
