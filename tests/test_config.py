class TestConfig:
    def test_class_labels(self):
        from src import config
        assert config.class_labels == ["minor", "moderate", "severe"]

    def test_img_shape(self):
        from src import config
        assert config.img_shape == (224, 224)

    def test_batch_size_positive(self):
        from src import config
        assert config.batch_size > 0

    def test_split_ratio_range(self):
        from src import config
        assert 0 < config.split_ratio < 1

    def test_paths_are_strings(self):
        from src import config
        for attr in ("data_dir", "models", "upload_path", "tuner_dir"):
            assert isinstance(getattr(config, attr), str), f"{attr} should be a string"