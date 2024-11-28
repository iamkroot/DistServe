from collections import namedtuple
from heapq import heappop, heapify
from typing import Generator

from distserve.request import Request
from distserve.logger import init_logger

logger = init_logger(__name__)


QueueElem = namedtuple("QueueElem", ["priority", "uniq_id", "req"])


class BoundedStarvationPriorityQueue:
    """
    Two level scheduler that uses priority to schedule jobs for the oldest chunk.
    """

    def __init__(self, pq_batch_size: int):
        self.pq_batch_size = pq_batch_size
        self.priority_queue: list[QueueElem] = []
        self.fcfs_queue: list[QueueElem] = []
    
    def add(self, req: Request):
        self.fcfs_queue.append(QueueElem(req.priority, id(req), req))
    
    def pop(self) -> Request:
        """Assumes len(self) > 0"""
        if not self.priority_queue:
            self._move_to_pq()
        return heappop(self.priority_queue).req
    
    def top(self) -> Request:
        """Assumes len(self) > 0"""
        if not self.priority_queue:
            self._move_to_pq()
        return self.priority_queue[0].req

    def _move_to_pq(self):
        self.priority_queue = self.fcfs_queue[:self.pq_batch_size]
        heapify(self.priority_queue)
        self.fcfs_queue = self.fcfs_queue[self.pq_batch_size:]
        logger.info(f"Put {len(self.priority_queue)} requests in the priority queue, remaining {len(self.fcfs_queue)} elements ")

    def __len__(self):
        return len(self.fcfs_queue) + len(self.priority_queue)
    
    def __bool__(self):
        return bool(self.fcfs_queue) or bool(self.priority_queue)
    
    def __str__(self):
        return f"PriorityQueue: {self.priority_queue}\nFCFSQueue: {self.fcfs_queue}"

    def __iter__(self) -> Generator[Request, None, None]:
        """Order NOT guaranteed to be priority based"""
        for elem in self.priority_queue:
            yield elem.req
        for elem in self.fcfs_queue:
            yield elem.req

