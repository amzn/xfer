import abc

from mxnet.gluon.data import DataLoader
import mxnet


class MetaTaskDataContainer(abc.ABC):

    def __init__(self, data_train, data_test, data_val, context=mxnet.cpu()):
        self.train_tasks = data_train
        self.test_tasks = data_test
        self.val_tasks = data_val
        self.context = context

    def get_train_tasks_iterators(self, batch_size, train=True):
        return self._get_tasks_iterators(self.train_tasks, batch_size, train=train)

    def get_val_tasks_iterators(self, batch_size, train=True):
        return self._get_tasks_iterators(self.val_tasks, batch_size, train=train)

    def get_test_tasks_iterators(self, batch_size, train=True):
        return self._get_tasks_iterators(self.test_tasks, batch_size, train=train)

    def _get_tasks_iterators(self, tasks, batch_size, train=True):
        if train is True:
            return [tt.get_train_iterator(batch_size) for tt in tasks]
        else:
            return [tt.get_val_iterator(batch_size) for tt in tasks]


class TaskDataContainer(abc.ABC):
    def __init__(self, train_dataset, val_dataset, context=mxnet.cpu()):
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self.context = context

    def get_train_val_iterators(self, batch_size):
        return {
            "train": self.get_train_iterator(batch_size),
            "val": self.get_val_iterator(batch_size)
        }

    def get_train_iterator(self, batch_size):
        return DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True)

    def get_val_iterator(self, batch_size):
        if self._val_dataset is not None:
            return DataLoader(self._val_dataset, batch_size=batch_size, shuffle=False)
        else:
            return None
