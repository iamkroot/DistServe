#  benchmark a version of QueueElem with NamedTuple instead of dataclass

from collections import namedtuple
from dataclasses import dataclass
from distserve.request import Request

QueueElemNT = namedtuple("QueueElemNT", ["priority", "uniq_id", "req"])

@dataclass(slots=True, frozen=True, order=True)
class QueueElemDC:
    priority: int
    uniq_id: int
    req: Request
    
    @classmethod
    def make(cls, req: Request):
        return cls(req.priority, id(req), req)

import time
import random
import numpy as np

from structs import Dataset
from pathlib import Path
EVAL_DIR = Path(__file__).parent.parent
assert EVAL_DIR.name == "evaluation"
DS_PATH = EVAL_DIR / "docs/datasets/sharegpt.ds"
dataset = Dataset.load(str(DS_PATH))
reqs = [Request(time.time(), i, treq.prompt, []) for i, treq in enumerate(dataset.reqs)]

from timeit import timeit

t0 = timeit("(req.priority, id(req), req)", setup="""
from distserve.request import Request
req = Request(0, 0, "ABC", [])
""", number=10000)
print(t0)

t1 = timeit("QueueElemNT(req.priority, id(req), req)", setup="""
from collections import namedtuple
QueueElemNT = namedtuple("QueueElemNT", ["priority", "uniq_id", "req"])
from distserve.request import Request
req = Request(0, 0, "ABC", [])
""", number=10000)
print(t1)

t2 = timeit("QueueElemDC(req.priority, id(req), req)", setup="""
from dataclasses import dataclass
from distserve.request import Request
@dataclass(slots=True, frozen=True, order=True)
class QueueElemDC:
    priority: int
    uniq_id: int
    req: Request
req = Request(0, 0, "ABC", [])
""", number=10000)
# t2 = timeit(lambda: [QueueElemDC.make(req) for req in reqs], number=1000)
print(t2)
# result: DC is 50% slower than NT
import sys
print(sys.getsizeof((0, 0, reqs[0])))
print(sys.getsizeof(QueueElemNT(0, 0, reqs[0])))
print(sys.getsizeof(QueueElemDC(0, 0, reqs[0])))
