import pytest

from main import Bandit


class TestBandit:
    def setup_method(self):
        self.bandit = Bandit(mean=0, stdev=1)

    def test_init(self):
        assert len(self.bandit) == 0
        assert len(self.bandit.observations) == 0
        assert len(self.bandit.observed_means) == 0
        with pytest.raises(IndexError):
            assert self.bandit.latest_mean == 0

    def test_first_step(self):
        self.bandit.step()
        assert len(self.bandit) == 1
        assert len(self.bandit.observations) == 1
        assert len(self.bandit.observed_means) == 1
