import random
import statistics

import numpy as np


class Bandit:
    """
    A gaussian random number generator.
    """

    def __init__(self, mean: float, stdev: float) -> None:
        self.true_mean: float = mean
        self.true_stdev: float = stdev
        self.observations: list[float] = []
        self.observed_means: list[float] = []

    def __len__(self) -> int:
        return len(self.observations)

    def __str__(self) -> str:
        return (
            f"n: {len(self.observations)} | "
            f"True Mean: {self.true_mean:.3f} | "
            f"True STDev: {self.true_stdev:.3f} | "
            f"Observed STDev: {statistics.stdev(self.observations):.3f}"
        )

    def step(self) -> float:
        observation = random.gauss(mu=self.true_mean, sigma=self.true_stdev)
        self.observations += [observation]
        if len(self.observations) == 1:
            next_mean = observation
        else:
            next_mean = incremental_mean(
                mean=self.observed_means[-1],
                observation=observation,
                n=len(self.observations) - 1,
            )
        self.observed_means += [next_mean]
        # TODO: Do I need to return anything?
        return observation


class TestBed:
    """
    A set of steps aginst a collection of Bandits.
    """

    def __init__(self, bandits: list[Bandit]) -> None:
        self.bandits: list[Bandit] = bandits
        self.observations: list[float] = []
        self.mean_history: list[float] = []
        self.mean: float = 0

    def __str__(self) -> str:
        return f"Bandits: {len(self.bandits)}, Steps: {len(self.observations)}"

    def best_bandit(self) -> Bandit:
        mean_list = [
            bandit.observed_means[-1] if len(bandit.observations) >= 1 else None
            for bandit in self.bandits
        ]
        if len(mean_list) == 0:
            bandit = random.choice(self.bandits)
        else:
            ## TODO: This is wrong. The list lengths are not the same.
            bandit = self.bandits[mean_list.index(max(mean_list))]
        return bandit

    #   def reset(self):
    #       # TODO: I think we can remove this?
    #       _ = [bandit.reset() for bandit in self.bandits]
    #       self.observations = []
    #       self.mean_history = []
    #       self.mean = 0

    def run_trials(self, steps: int, epsilon: float):
        for step in range(steps):
            if random.uniform(0, 1) <= epsilon:
                bandit = random.choice(self.bandits)
            else:
                bandit = self.best_bandit()
            self.observations += [bandit.step()]
            if step == 0:
                self.mean_history += [self.observations[0]]
            else:
                self.mean_history += [
                    incremental_mean(
                        self.mean_history[-1],
                        self.observations[-1],
                        len(self.observations) - 1,
                    )
                ]

        # TODO: Should we do this incrementally within the for loop?
        # self.mean_history = rolling_incremental_mean(self.observations)
        # return self.observations, self.bandits


def incremental_mean(mean: float, observation: float, n: int) -> float:
    return mean + ((observation - mean) / n)


def rolling_incremental_mean(observations: list[float]) -> list[float]:
    rolling_mean: list[float] = []
    for counter, observation in enumerate(observations):
        if counter == 0:
            rolling_mean += [observation]
        else:
            rolling_mean += [
                incremental_mean(
                    mean=rolling_mean[-1], observation=observation, n=counter + 1
                )
            ]
    return rolling_mean


def run_experiments(bandit_count: int, steps: int, experiments: int, epsilon: float):
    container = np.empty((experiments, steps))
    for counter in range(experiments):
        bandits = [
            Bandit(mean=random.gauss(mu=0, sigma=1), stdev=1)
            for _ in range(bandit_count)
        ]
        test_bed = TestBed(bandits)

        test_bed.run_trials(steps=steps, epsilon=epsilon)
        # results, bandits_output = test_bed.run_trials(steps=steps, epsilon=epsilon)
        # TODO: I think I can replace this with a call to TestBed.mean_history
        # means = rolling_incremental_mean(results)
        # test_bed.reset()
        # container[counter, :] = means
        container[counter, :] = test_bed.mean_history
    return container.mean(axis=0)


# TODO: If this is a library we can remove this
if __name__ == "__main__":
    results_010 = run_experiments(
        bandit_count=10, steps=1000, experiments=2000, epsilon=0.1
    )
    results_001 = run_experiments(
        bandit_count=10, steps=1000, experiments=2000, epsilon=0.01
    )
    results_000 = run_experiments(
        bandit_count=10, steps=1000, experiments=2000, epsilon=0.00
    )
