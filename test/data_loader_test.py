import os
import unittest

import h5py
import torch

from pointcountfm.data_loader import DataLoader
from pointcountfm.preprocessing import Identity


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_file = "test_data.h5"
        self.batch_size = 2
        self.shuffle = False

        # Create a mock HDF5 file with test data
        with h5py.File(self.data_file, "w") as f:
            f.create_dataset("energy", data=[1.0, 2.0, 3.0, 4.0])
            f.create_dataset(
                "num_points",
                data=[[13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
            )

    def tearDown(self):
        os.remove(self.data_file)

    def test_data_loader_initialization(self):
        data_loader = DataLoader(
            data_file=self.data_file,
            transform_inc=Identity(),
            transform_num_points=Identity(),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        self.assertEqual(data_loader.num_samples, 4)
        self.assertEqual(len(data_loader), 2)

    def test_data_loader_iteration(self):
        data_loader = DataLoader(
            data_file=self.data_file,
            transform_inc=Identity(),
            transform_num_points=Identity(),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        batches = list(data_loader)
        self.assertEqual(len(batches), 2)
        self.assertTrue(
            torch.equal(batches[0]["data"], torch.tensor([[13.0, 14.0], [15.0, 16.0]]))
        )
        self.assertTrue(torch.equal(batches[0]["condition"], torch.tensor([1.0, 2.0])))

    def test_data_loader_shuffle(self):
        data_loader = DataLoader(
            data_file=self.data_file,
            transform_inc=Identity(),
            transform_num_points=Identity(),
            batch_size=self.batch_size,
            shuffle=True,
        )
        batches = list(data_loader)
        self.assertEqual(len(batches), 2)


if __name__ == "__main__":
    unittest.main()
