import pytest
import time
import os

class TestDamageScore:
    def test_pure_minor(self):
        from src.predictor import damage_score
        score = damage_score(1.0, 0.0, 0.0)
        assert score == 2.5

    def test_pure_moderate(self):
        from src.predictor import damage_score
        score = damage_score(0.0, 1.0, 0.0)
        assert score == 5.5

    def test_pure_severe(self):
        from src.predictor import damage_score
        score = damage_score(0.0, 0.0, 1.0)
        assert score == 9.0

    def test_mixed(self):
        from src.predictor import damage_score
        score = damage_score(1 / 3, 1 / 3, 1 / 3)
        expected = round((2.5 + 5.5 + 9.0) / 3, 1)
        assert score == expected

    def test_score_in_range(self):
        from src.predictor import damage_score
        import random
        rng = random.Random(42)
        for _ in range(20):
            a, b = rng.random(), rng.random()
            c = 1 - a - b
            if c < 0:
                a, b, c = 0.5, 0.3, 0.2
            assert 0.0 <= damage_score(a, b, c) <= 10.0


class TestMajorityVote:
    def test_single_model(self):
        import numpy as np
        from src.ensemble import majority_vote
        probs = [np.array([0.1, 0.2, 0.7])]
        label, conf = majority_vote(probs)
        assert label == "severe"
        assert pytest.approx(conf, abs=1e-6) == 0.7

    def test_consensus(self):
        import numpy as np
        from src.ensemble import majority_vote
        probs = [
            np.array([0.8, 0.1, 0.1]),
            np.array([0.7, 0.2, 0.1]),
            np.array([0.9, 0.05, 0.05]),
        ]
        label, conf = majority_vote(probs)
        assert label == "minor"
        assert conf > 0.7

    def test_split_vote_takes_mean(self):
        """With two opposing votes the mean should decide."""
        import numpy as np
        from src.ensemble import majority_vote
        probs = [
            np.array([0.9, 0.05, 0.05]),
            np.array([0.05, 0.9, 0.05]),
        ]
        label, _ = majority_vote(probs)
        assert label in ("minor", "moderate")

    def test_returns_valid_label(self):
        import numpy as np
        from src import config
        from src.ensemble import majority_vote
        probs = [np.array([0.33, 0.33, 0.34])]
        label, conf = majority_vote(probs)
        assert label in config.class_labels
        assert 0.0 <= conf <= 1.0

class TestRerank:
    def test_rerank_adds_final_rank_column(self):
        import pandas as pd
        from src.ensemble import rerank_combinations

        data = {
            "accuracy":        [0.9, 0.85, 0.8],
            "mean_confidence": [0.88, 0.80, 0.75],
            "combo_size":      [2, 3, 1],
        }
        df = pd.DataFrame(data)
        out = rerank_combinations(df)
        assert "final_rank" in out.columns
        assert out["final_rank"].min() == 1

    def test_smaller_better_size_ranks_first_when_equal_accuracy(self):
        import pandas as pd
        from src.ensemble import rerank_combinations

        data = {
            "accuracy":        [0.9, 0.9],
            "mean_confidence": [0.9, 0.9],
            "combo_size":      [3, 1],
        }
        df = pd.DataFrame(data)
        out = rerank_combinations(df)
        rank_large = out.loc[out["combo_size"] == 3, "final_rank"].iloc[0]
        rank_small = out.loc[out["combo_size"] == 1, "final_rank"].iloc[0]
        assert rank_small <= rank_large

class TestClearOldUploads:
    def test_removes_old_files(self, tmp_path):
        from utils.utils import clear_old_uploads
        old_file = tmp_path / "old_upload.jpg"
        old_file.write_bytes(b"fake image data")
        old_time = time.time() - (8 * 24 * 3600)
        os.utime(old_file, (old_time, old_time))

        clear_old_uploads(str(tmp_path), max_age_days=7)
        assert not old_file.exists()

    def test_keeps_recent_files(self, tmp_path):
        from utils.utils import clear_old_uploads
        new_file = tmp_path / "new_upload.jpg"
        new_file.write_bytes(b"fake image data")
        # mtime defaults to now

        clear_old_uploads(str(tmp_path), max_age_days=7)
        assert new_file.exists()

    def test_ignores_directories(self, tmp_path):
        from utils.utils import clear_old_uploads
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        old_time = time.time() - (8 * 24 * 3600)
        os.utime(subdir, (old_time, old_time))

        clear_old_uploads(str(tmp_path), max_age_days=7)
        assert subdir.exists()

    def test_empty_directory_is_fine(self, tmp_path):
        from utils.utils import clear_old_uploads
        clear_old_uploads(str(tmp_path))