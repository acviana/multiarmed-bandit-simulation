import random
import statistics

import numpy as np


class Bandit:
    """
    A gaussian random number generator with history of pervious values and means.
    """

    def __init__(self, mean: float, stdev: float) -> None:
        self.true_mean: float = mean
        self.true_stdev: float = stdev
        self.observations: list[float] = []
        self.means: list[float] = []

    def __len__(self) -> int:
        return len(self.observations)

    def __str__(self) -> str:
        return (
            f"n: {len(self.observations)} | "
            f"Observed Mean: {self.means[-1]:.3f} | "
            f"True Mean: {self.true_mean:.3f} | "
            f"Observed STDev: {statistics.stdev(self.observations):.3f} | "
            f"True STDev: {self.true_stdev:.3f}"
        )

    def step(self) -> float:
        """Draw one observation and update object attributes."""
        observation: float = random.gauss(mu=self.true_mean, sigma=self.true_stdev)
        self.observations += [observation]
        next_mean: float
        if len(self.observations) == 1:
            next_mean = observation
        else:
            next_mean = incremental_mean(
                mean=self.means[-1],
                observation=observation,
                n=len(self.observations) - 1,
            )
        self.means += [next_mean]
        return observation


class TestBed:
    """
    A set of steps aginst a collection of Bandits.
    """

    def __init__(
        self,
        bandits: list[Bandit],
        default_mean: float,
    ) -> None:
        self.bandits: list[Bandit] = bandits
        self.observations: list[float] = []
        self.means: list[float] = []
        self.default_mean: float = default_mean

    def __str__(self) -> str:
        return (
            f"Bandits: {len(self.bandits)} | "
            f"Steps: {len(self.observations)} | "
            f"Average Reward: {self.means[-1]}"
        )

    def __len__(self) -> int:
        return len(self.observations)

    def best_bandit(self) -> Bandit:
        """Return the bandit with the highest mean reward."""
        bandit_dict: dict[float, int] = {
            (bandit.means[-1] if len(bandit.means) > 0 else self.default_mean): index
            for index, bandit in enumerate(self.bandits)
        }
        bandit = self.bandits[bandit_dict[max(bandit_dict.keys())]]
        if len(bandit_dict) == 0:
            bandit = random.choice(self.bandits)
        return bandit

    def run_trials(self, steps: int, epsilon: float):
        """Run a trial picking either the best bandit or a random bandit with probabily epsilon."""
        bandit: Bandit
        for step in range(steps):
            if random.uniform(0, 1) <= epsilon:
                bandit = random.choice(self.bandits)
            else:
                bandit = self.best_bandit()
            self.observations += [bandit.step()]
            if step == 0:
                self.means += [self.observations[0]]
            else:
                self.means += [
                    incremental_mean(
                        self.means[-1],
                        self.observations[-1],
                        len(self.observations) - 1,
                    )
                ]


def incremental_mean(mean: float, observation: float, n: int) -> float:
    """Efficient incrmental mean calculator.

    Returns the mean at n+1 given the current mean, n, and the next
    observation.
    """
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
        test_bed = TestBed(bandits=bandits, default_mean=0.0)

        test_bed.run_trials(steps=steps, epsilon=epsilon)
        # results, bandits_output = test_bed.run_trials(steps=steps, epsilon=epsilon)
        # TODO: I think I can replace this with a call to TestBed.means
        # means = rolling_incremental_mean(results)
        # test_bed.reset()
        # container[counter, :] = means
        container[counter, :] = test_bed.means
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
