from contextlib import contextmanager
import traceback
from typing import Any

from distributed import Client, get_worker, wait, worker_client
from kedro.io.core import AbstractDataSet


@contextmanager
def maybe_worker_client():
    try:
        get_worker()
    except ValueError:
        yield Client.current()
    else:
        # this is wierd - I don't understand why this is necessary
        with worker_client(separate_thread=False) as c:
            yield c


class DaskDataset(AbstractDataSet):
    def __init__(self, name):
        self.name = name

    def _load(self) -> Any:
        with worker_client() as c:
            return c.get_dataset(self.name)

    def _save(self, data: Any):
        with maybe_worker_client() as c:
            c.publish_dataset(data, name=self.name, override=True)

    def _describe(self):
        return dict(name=self.name)
