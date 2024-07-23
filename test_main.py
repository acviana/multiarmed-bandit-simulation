import pytest

from main import incremental_mean, Bandit, TestBed


def test_incremental_mean():
    assert incremental_mean(mean=0.0, observation=1.0, n=2) == 0.5
    assert incremental_mean(mean=0.5, observation=1.0, n=3) == 2.0 / 3.0
    assert incremental_mean(mean=(2.0 / 3.0), observation=2.0, n=4) == 1.0


class TestBandit:
    def setup_method(self):
        self.bandit = Bandit(mean=0, stdev=1)

    def test_init(self):
        assert len(self.bandit) == 0
        assert len(self.bandit.observations) == 0
        assert len(self.bandit.means) == 0
        with pytest.raises(IndexError):
            assert self.bandit.means[-1] == 0

    def test_first_step(self):
        self.bandit.step()
        assert len(self.bandit) == 1
        assert len(self.bandit.observations) == 1
        assert len(self.bandit.means) == 1


class TestTestBed:
    def setup_method(self):
        self.test_bed = TestBed(
            bandits=[Bandit(mean=0, stdev=1) for _ in range(10)], default_mean=0
        )

    def test_best_bandit(self):
        self.test_bed.best_bandit()

    def test_run_trials(self):
        self.test_bed.run_trials(steps=10, epsilon=0.1)
