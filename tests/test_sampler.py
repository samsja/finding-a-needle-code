from src.few_shot_learning import RelationNet,FewShotSampler
from src.few_shot_learning.datasets import FewShotDataSet 

import torch

import unittest


class RandomFSDataSet(FewShotDataSet):
    
    def __init__(self):
        super().__init__()
        
        self._classes = torch.arange(10)
        self._length_of_class = 20
    
    def __getitm__(self,idx):
        return ( torch.zeros((10,10)),0)

    def get_index_in_class(self, class_idx: int):
        """
        Method to get the indexes of the elements in the same class as class_idx

        # Args:
            class_idx : int. The index of the desider class

        """
        return torch.arange(
            class_idx * self._length_of_class,
            (class_idx + 1) * self._length_of_class,
            dtype=torch.long,
        )



class TestFewShotSampler(unittest.TestCase):
    def setUp(self):

        self.ep = 10
        self.n = 2
        self.k = 5
        self.q = 15
        self.f_dim = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.dataset = RandomFSDataSet()
        self.few_shot_sampler = FewShotSampler(
            self.dataset,
            number_of_batch=2,
            episodes=self.ep,
            sample_per_class=self.n,
            classes_per_ep=self.k,
            queries=self.q,
        )

    def test_output_shape(self):
        assert next(iter(self.few_shot_sampler)).shape == (self.ep*self.k*(self.n+self.q),1) 