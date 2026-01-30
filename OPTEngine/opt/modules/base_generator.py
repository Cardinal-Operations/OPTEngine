from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    def __init__(self, samples_per_type=10, seed=42):
        self.samples_per_type = samples_per_type
        self.seed = seed

    @abstractmethod
    def generate_instances(self, output_path):
        pass

    @abstractmethod
    def map_to_nl(self, input_path, output_path):
        pass