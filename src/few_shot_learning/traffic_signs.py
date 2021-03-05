import torch
import torchvision
from torchvision import transforms

from .datasets import FewShotDataSet


class TrafficSignDataset(FewShotDataSet):
    def __init__(self, *args, **kwargs):
        super(TrafficSignDataset, self).__init__(*args, **kwargs)

        def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the folders.
        """
        self.data = []
        self.labels = []
        
        self.root = root_dir

        for i, c in enumerate(os.listdir(root_dir)):
            list_of_datapoints = os.listdir(root_dir+"/"+c)
            self.data += list_of_datapoints
            self.labels += [i]*len(list_of_datapoints)

        self._classes = list(set(self.labels))

        self.transform = transforms.Compose([
                                            transforms.Rescale(256),
                                            transforms.RandomCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                           ]))

    def get_index_in_class(self, class_idx: int):
        """
        Method to get the indexes of the elements in the same class as class_idx

        # Args:
            class_idx : int. The index of the desider class

        """

        start = self.labels.index(class_idx)
        try:
            end = self.labels.index(class_idx+1)
        except:
            end = len(self.labels.index)

        return torch.arange(start, end)

    def __getitem__(self, idx):
        x = Image.open(self.data[idx])
        x = self.transform(x)

        y = torch.tensor(self.labels[idx])

        return x, y

