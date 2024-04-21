    import random
import statistics


class Bandit:
    def __init__(self, mean: float, stdev: float) -> None:
        self.true_mean = mean
        self.true_stdev = stdev
        self.observations: list[float] = []
        self.observed_mean: float = 0

    def __str__(self):
        return (
            f"n: {len(self.observations)} | "
            f"True Mean: {self.true_mean:.3f} | "
            f"Observed Mean: {self.observed_mean:.3f} | "
            f"True STDev: {self.true_stdev:.3f} | "
            f"Observed STDev: {statistics.stdev(self.observations):.3f}"
        )

    def reset(self) -> None:
        self.observations = []
        self.observed_mean = 0

    def step(self) -> float:
        observation = random.gauss(mu=self.true_mean, sigma=self.true_stdev)
        self.observations += [observation]
        self.observed_mean = incremental_mean(
            mean=self.observed_mean, observation=observation, n=len(self.observations)
        )
        return observation


class TestBed:
    def __init__(self, bandits: list[Bandit]) -> None:
        self.bandits = bandits
        self.observations: list[float] = []
        self.mean_history: list[float] = []
        self.mean: float = 0

    def best_bandit(self) -> Bandit:
        mean_list = [bandit.observed_mean for bandit in bandits]
        return bandits[mean_list.index(max(mean_list))]

    def reset(self):
        _ = [bandit.reset() for bandit in bandits]
        self.observations = []
        self.mean_history = []
        self.mean = 0

    def run_trial(self, steps: int, epsilon: float):
        for _ in range(steps):
            if random.uniform(0, 1) <= epsilon:
                bandit = random.choice(self.bandits)
            else:
                bandit = self.best_bandit()
            self.observations += [bandit.step()]
        return self.observations, self.bandits


def incremental_mean(mean: float, observation: float, n: int) -> float:
    return mean + ((observation - mean) / n)


def rolling_incremental_mean(observations: list[float]) -> list[float]:
    rolling_mean: list[float] = []
    for counter, observation in enumerate(observations):
        if counter == 0:
            mean = 0
        mean = incremental_mean(mean=mean, observation=observation, n=counter + 1)
        rolling_mean += [mean]
    return rolling_mean


def run_experiments(bandit_count: int, trials: int, experiments: int, epsilon: float):
    container = np.empty((experiments, trials))
    for counter in range(experiments):
        bandits = [Bandit(mean=random.gauss(), stdev=1) for _ in range(bandit_count)]
        test_bed = TestBed(bandits)

        results, bandits_output = test_bed.run_trials(trials=trials, epsilon=epsilon)
        means = rolling_incremental_mean(results)
        test_bed.reset()
        container[counter, :] = means
    return container.mean(axis=0)


if __name__ == "__main__":
    results_010 = run_experiments(
        bandit_count=10, trials=1000, experiments=2000, epsilon=0.1
    )
    results_001 = run_experiments(
        bandit_count=10, trials=1000, experiments=2000, epsilon=0.01
    )
    results_000 = run_experiments(
        bandit_count=10, trials=1000, experiments=2000, epsilon=0.00
    )
