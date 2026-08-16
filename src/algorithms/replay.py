import random

from torch.utils.data import ConcatDataset, DataLoader, Subset


class ReplayBuffer:
    def __init__(self, memory_size=1000, seed=42):
        if memory_size < 1:
            raise ValueError("memory_size must be at least 1")

        self.memory_size = memory_size
        self.seed = seed
        self.task_memories = []

    def add_task(self, dataset):
        rng = random.Random(
            self.seed + len(self.task_memories)
        )

        num_samples = min(
            self.memory_size,
            len(dataset)
        )

        indices = rng.sample(
            range(len(dataset)),
            num_samples
        )

        memory = Subset(
            dataset,
            indices
        )

        self.task_memories.append(memory)

    def create_replay_loader(
        self,
        current_dataset,
        batch_size=64
    ):
        datasets = [current_dataset]

        datasets.extend(
            self.task_memories
        )

        combined_dataset = ConcatDataset(
            datasets
        )

        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True
        )