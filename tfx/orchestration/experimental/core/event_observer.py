# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""event_observer is a module for registering observers to observe events.

event_observer.init() must be called exactly once before any
event_observer.get() calls.

event_observer.shutdown() must be called before the program shuts down,
otherwise the EventObserver thread may not exit and the program may fail to
terminate, and be stuck waiting forever for the EventObserver thread to finish.
"""

from concurrent import futures
import dataclasses
import queue
import threading
from typing import Callable, List, Union

from absl import logging
from ml_metadata.proto import metadata_store_pb2


@dataclasses.dataclass(frozen=True)
class PipelineStarted:
  execution: metadata_store_pb2.Execution
  pipeline_id: str


@dataclasses.dataclass(frozen=True)
class PipelineFinished:
  execution: metadata_store_pb2.Execution
  pipeline_id: str


@dataclasses.dataclass(frozen=True)
class NodeStateChange:
  execution: metadata_store_pb2.Execution
  pipeline_id: str
  pipeline_run: str
  node_id: str
  old_state: str
  new_state: str


Event = Union[PipelineStarted, PipelineFinished, NodeStateChange]

ObserverFn = Callable[[Event], None]


def register_observer(observer_fn: ObserverFn):
  """Register an observer.

  The observer function will be called whenever an event triggers.

  Args:
    observer_fn: A function that takes in an ObserverEvent.
  """
  get().register_observer(observer_fn)


# Users should not need to use methods below this line.


class DoNothingEventObserver:
  """EventObserver that does nothing.

  This class exists so that other modules or tests which depend on
  PipelineState but don't care about the event observation don't need to
  initialise EventObserver.
  """

  def shutdown(self) -> None:
    pass

  def register_observer(self, observer_fn: ObserverFn) -> None:
    pass

  def notify(self, event: Event) -> None:
    pass

  def testonly_wait(self) -> None:
    pass


class EventObserver(DoNothingEventObserver):
  """EventObserver.

  Users should only call the module-level register_observer function. Only
  orchestrator-internal code will interact with this class directly.

  Events are guaranteed to be observed in the order they were notified.

  Observer functions *may* be called in any order (even though the current
  implementation calls them in the registration order, this may change).

  Observer functions *may* be called concurrently (even though the current
  implementation calls them serially, this may change).

  Exceptions in the observer functions are logged, but ignored. Note that a
  slow or stuck observer function may cause events to stop getting observed
  (which is why we may switch to calling them concurrently / with a timeout
  in the future).
  """
  _event_queue: queue.Queue
  _observers: List[ObserverFn]
  _executor: futures.ThreadPoolExecutor

  def __init__(self):
    """EventObserver constructor."""
    self._event_queue = queue.Queue()
    self._observers = []
    self._shutdown_event = threading.Event()
    self._main_executor = futures.ThreadPoolExecutor(max_workers=1)
    self._main_future = None

  def shutdown(self) -> None:
    self._shutdown_event.set()
    self._main_executor.shutdown()

  def register_observer(self, observer_fn: ObserverFn) -> None:
    self._observers.append(observer_fn)

  def notify(self, event: Event) -> None:
    if not self._observers:
      return
    if not self._main_future:
      # We have to start the main thread lazily here, and not in the
      # constructor, since in the constructor the object hasn't been fully
      # initialised.
      self._main_future = self._main_executor.submit(self._main)
    self._event_queue.put(event)

  def testonly_wait(self) -> None:
    """Wait for all existing events in the queue to be observed.

    For use in tests only.
    """
    self._event_queue.join()

  def _main(self) -> None:
    """Main observation loop. Checks event queue for events, calls observers."""

    def observe_event(event):
      for observer_fn in self._observers:
        try:
          observer_fn(event)
        except Exception as e:  # pylint: disable=broad-except
          logging.exception(
              'Exception raised by observer function when observing '
              'event %s: %s', event, e)

    def dequeue():
      try:
        event = self._event_queue.get(block=True, timeout=5)
        return event
      except queue.Empty:
        return None

    while not self._shutdown_event.is_set():
      event = dequeue()
      if event is not None:
        observe_event(event)
        self._event_queue.task_done()


_event_observer = None


def init() -> None:
  """Initialises the singleton EventObserver."""
  global _event_observer
  if _event_observer:
    raise RuntimeError('init() was already called')
  _event_observer = EventObserver()


def get() -> Union[DoNothingEventObserver, EventObserver]:
  """Returns the singleton EventObserver."""
  global _event_observer
  if not _event_observer:
    return DoNothingEventObserver()
  return _event_observer


def shutdown() -> None:
  """Shutdowns and cleans up the singleton EventObserver."""
  global _event_observer
  _event_observer.shutdown()
  _event_observer = None
