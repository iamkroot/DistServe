from collections import namedtuple
from heapq import heappop, heapify, heappush
from typing import Generator, Literal, TypeVar, Generic, NamedTuple

from distserve.request import Request, MigratingRequest
from distserve.logger import init_logger

logger = init_logger(__name__)


ReqT = TypeVar('ReqT', Request, MigratingRequest)
# TODO: Make QueueElem generic over request type instead of using typevar
QueueElem = NamedTuple("QueueElem", priority=int, uniq_id=int, req=ReqT)

class BoundedStarvationPriorityQueue(Generic[ReqT]):
    """
    Two level scheduler that uses priority to schedule jobs for the oldest chunk.
    """

    def __init__(self, pq_batch_size: int, mode: Literal['context', 'decoding']):
        self.pq_batch_size = pq_batch_size
        self.priority_queue: list[QueueElem] = []
        self.fcfs_queue: list[QueueElem] = []
        self.mode = mode
    
    def add(self, req: ReqT):
        self.fcfs_queue.append(QueueElem(req.priority if isinstance(req, Request) else req.req.priority, id(req), req))
    
    def pop(self) -> ReqT:
        """Assumes len(self) > 0"""
        if not self.priority_queue:
            self._move_to_pq()
        return heappop(self.priority_queue).req
    
    def top(self) -> ReqT:
        """Assumes len(self) > 0"""
        if not self.priority_queue:
            self._move_to_pq()
        return self.priority_queue[0].req

    def _move_to_pq(self):
        self.priority_queue = self.fcfs_queue[:self.pq_batch_size]
        heapify(self.priority_queue)
        self.fcfs_queue = self.fcfs_queue[self.pq_batch_size:]
        logger.info(f"({self.mode}) Put {len(self.priority_queue)} requests in the priority queue, remaining {len(self.fcfs_queue)} elements ")

    def __len__(self):
        return len(self.fcfs_queue) + len(self.priority_queue)
    
    def __bool__(self):
        return bool(self.fcfs_queue) or bool(self.priority_queue)
    
    def __str__(self):
        return f"({self.mode}) PriorityQueue: {self.priority_queue}\nFCFSQueue: {self.fcfs_queue}"

    def __iter__(self) -> Generator[ReqT, None, None]:
        """Order NOT guaranteed to be priority based"""
        for elem in self.priority_queue:
            yield elem.req
        for elem in self.fcfs_queue:
            yield elem.req


class DirectPriorityQueue(Generic[ReqT]):
    """
    Wrapper over heap
    """

    def __init__(self, mode: Literal['context', 'decoding']):
        self.priority_queue: list[QueueElem] = []
        self.mode = mode
    
    def add(self, req: ReqT):
        heappush(self.priority_queue, QueueElem(req.priority if isinstance(req, Request) else req.req.priority, id(req), req))
    
    def pop(self) -> ReqT:
        """Assumes len(self) > 0"""
        return heappop(self.priority_queue).req
    
    def top(self) -> ReqT:
        """Assumes len(self) > 0"""
        return self.priority_queue[0].req

    def __len__(self):
        return len(self.priority_queue)
    
    def __bool__(self):
        return bool(self.priority_queue)
    
    def __str__(self):
        return f"({self.mode}) PriorityQueue: {self.priority_queue}"

    def __iter__(self) -> Generator[ReqT, None, None]:
        """Order NOT guaranteed to be priority based"""
        for elem in self.priority_queue:
            yield elem.req

