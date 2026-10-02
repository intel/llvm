//==------------------------- AsyncAlloc.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Tests that the queue overloads of the asynchronous allocation functions are
// submitted directly to the backend, without going through the handler, and
// without requesting events which nobody can observe.

#include <helpers/UrMock.hpp>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

#include <gtest/gtest.h>

using namespace sycl;
namespace oneapiext = ext::oneapi::experimental;

namespace {

size_t CounterAlloc = 0;
size_t CounterFree = 0;
size_t CounterEventsWait = 0;
size_t CounterBarrier = 0;
// Number of allocation/free calls which requested an output event from the
// backend.
size_t CounterAllocWithEvent = 0;
size_t CounterFreeWithEvent = 0;

ur_result_t redefined_urEnqueueUSMDeviceAllocExp(void *pParams) {
  auto Params =
      *static_cast<ur_enqueue_usm_device_alloc_exp_params_t *>(pParams);
  ++CounterAlloc;
  if (*Params.pphEvent)
    ++CounterAllocWithEvent;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefined_urEnqueueUSMFreeExp(void *pParams) {
  auto Params = *static_cast<ur_enqueue_usm_free_exp_params_t *>(pParams);
  ++CounterFree;
  if (*Params.pphEvent)
    ++CounterFreeWithEvent;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefined_urEnqueueEventsWait(void *) {
  ++CounterEventsWait;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefined_urEnqueueEventsWaitWithBarrierExt(void *) {
  ++CounterBarrier;
  return UR_RESULT_SUCCESS;
}

// Set by the host task and checked by the free callback in the
// InOrderQueueShortcutAfterHostTask test below.
std::atomic<bool> HostTaskExecuted = false;

ur_result_t redefined_urEnqueueUSMFreeExpAfterHostTask(void *pParams) {
  EXPECT_TRUE(HostTaskExecuted.load());
  return redefined_urEnqueueUSMFreeExp(pParams);
}

class AsyncAllocTests : public ::testing::Test {
protected:
  void SetUp() override {
    CounterAlloc = 0;
    CounterFree = 0;
    CounterEventsWait = 0;
    CounterBarrier = 0;
    CounterAllocWithEvent = 0;
    CounterFreeWithEvent = 0;

    mock::getCallbacks().set_replace_callback(
        "urEnqueueUSMDeviceAllocExp", &redefined_urEnqueueUSMDeviceAllocExp);
    mock::getCallbacks().set_replace_callback("urEnqueueUSMFreeExp",
                                              &redefined_urEnqueueUSMFreeExp);
    mock::getCallbacks().set_replace_callback("urEnqueueEventsWait",
                                              &redefined_urEnqueueEventsWait);
    mock::getCallbacks().set_replace_callback(
        "urEnqueueEventsWaitWithBarrierExt",
        &redefined_urEnqueueEventsWaitWithBarrierExt);
  }

  queue makeQueue(bool InOrder) {
    return InOrder ? queue{context(platform()),
                           default_selector_v,
                           {property::queue::in_order{}}}
                   : queue{context(platform()), default_selector_v};
  }

  unittest::UrMock<> Mock;
};

// The queue overloads do not return an event, so on an in-order queue no
// backend event has to be created at all.
TEST_F(AsyncAllocTests, InOrderQueueShortcutNoEvents) {
  queue Q = makeQueue(/*InOrder=*/true);

  void *Ptr = oneapiext::async_malloc(Q, usm::alloc::device, 1024);
  oneapiext::async_free(Q, Ptr);

  EXPECT_EQ(CounterAlloc, size_t{1});
  EXPECT_EQ(CounterFree, size_t{1});
  EXPECT_EQ(CounterAllocWithEvent, size_t{0});
  EXPECT_EQ(CounterFreeWithEvent, size_t{0});

  // In-order queues are ordered by the backend, so no additional
  // synchronization commands are needed.
  EXPECT_EQ(CounterEventsWait, size_t{0});
  EXPECT_EQ(CounterBarrier, size_t{0});
}

TEST_F(AsyncAllocTests, InOrderQueueShortcutFromPoolNoEvents) {
  queue Q = makeQueue(/*InOrder=*/true);
  auto Pool = Q.get_context().ext_oneapi_get_default_memory_pool(
      Q.get_device(), usm::alloc::device);

  void *Ptr = oneapiext::async_malloc_from_pool(Q, 1024, Pool);
  oneapiext::async_free(Q, Ptr);

  EXPECT_EQ(CounterAlloc, size_t{1});
  EXPECT_EQ(CounterFree, size_t{1});
  EXPECT_EQ(CounterAllocWithEvent, size_t{0});
  EXPECT_EQ(CounterFreeWithEvent, size_t{0});
  EXPECT_EQ(CounterEventsWait, size_t{0});
  EXPECT_EQ(CounterBarrier, size_t{0});
}

// Repeated allocations must not accumulate any synchronization commands: the
// queue has to stay in the "no last event" mode.
TEST_F(AsyncAllocTests, InOrderQueueShortcutRepeated) {
  queue Q = makeQueue(/*InOrder=*/true);
  constexpr size_t Iterations = 4;

  for (size_t I = 0; I < Iterations; ++I) {
    void *Ptr = oneapiext::async_malloc(Q, usm::alloc::device, 1024);
    oneapiext::async_free(Q, Ptr);
  }

  EXPECT_EQ(CounterAlloc, Iterations);
  EXPECT_EQ(CounterFree, Iterations);
  EXPECT_EQ(CounterAllocWithEvent, size_t{0});
  EXPECT_EQ(CounterFreeWithEvent, size_t{0});
  EXPECT_EQ(CounterEventsWait, size_t{0});
  EXPECT_EQ(CounterBarrier, size_t{0});
}

// Out-of-order queues provide no implicit ordering which would make the event
// of a submission redundant, so an event is requested from the backend even
// though the user cannot observe it. This matches the other submission paths
// which bypass the scheduler, all of which only discard events on in-order
// queues.
TEST_F(AsyncAllocTests, OutOfOrderQueueShortcutEvents) {
  queue Q = makeQueue(/*InOrder=*/false);

  void *Ptr = oneapiext::async_malloc(Q, usm::alloc::device, 1024);
  oneapiext::async_free(Q, Ptr);

  EXPECT_EQ(CounterAlloc, size_t{1});
  EXPECT_EQ(CounterFree, size_t{1});
  EXPECT_EQ(CounterAllocWithEvent, size_t{1});
  EXPECT_EQ(CounterFreeWithEvent, size_t{1});
}

// The handler overloads have to produce an event, as it can be obtained
// through submit_with_event.
TEST_F(AsyncAllocTests, HandlerOverloadEvents) {
  queue Q = makeQueue(/*InOrder=*/true);

  void *Ptr = nullptr;
  event AllocEvent = oneapiext::submit_with_event(Q, [&](handler &CGH) {
    Ptr = oneapiext::async_malloc(CGH, usm::alloc::device, 1024);
  });
  oneapiext::submit(Q, [&](handler &CGH) {
    CGH.depends_on(AllocEvent);
    oneapiext::async_free(CGH, Ptr);
  });

  EXPECT_EQ(CounterAlloc, size_t{1});
  EXPECT_EQ(CounterFree, size_t{1});
  EXPECT_EQ(CounterAllocWithEvent, size_t{1});
}

// The allocation is enqueued by async_malloc() itself, so a dependency added
// afterwards would silently have no effect and is rejected instead.
TEST_F(AsyncAllocTests, HandlerOverloadDependsOnAfterAlloc) {
  queue Q = makeQueue(/*InOrder=*/true);

  event Event = Q.ext_oneapi_submit_barrier();

  try {
    oneapiext::submit(Q, [&](handler &CGH) {
      oneapiext::async_malloc(CGH, usm::alloc::device, 1024);
      CGH.depends_on(Event);
      FAIL() << "Expected an exception.";
    });
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::errc::invalid);
  }
}

// The handler overloads submitted without an event must not request one from
// the backend either, as nothing would take ownership of it.
TEST_F(AsyncAllocTests, HandlerOverloadNoEvents) {
  queue Q = makeQueue(/*InOrder=*/true);
  auto Pool = Q.get_context().ext_oneapi_get_default_memory_pool(
      Q.get_device(), usm::alloc::device);
  constexpr size_t Iterations = 4;

  for (size_t I = 0; I < Iterations; ++I) {
    void *Ptr = nullptr;
    oneapiext::submit(Q, [&](handler &CGH) {
      Ptr = oneapiext::async_malloc(CGH, usm::alloc::device, 1024);
    });
    oneapiext::submit(Q,
                      [&](handler &CGH) { oneapiext::async_free(CGH, Ptr); });

    void *PoolPtr = nullptr;
    oneapiext::submit(Q, [&](handler &CGH) {
      PoolPtr = oneapiext::async_malloc_from_pool(CGH, 1024, Pool);
    });
    oneapiext::submit(
        Q, [&](handler &CGH) { oneapiext::async_free(CGH, PoolPtr); });
  }

  EXPECT_EQ(CounterAlloc, 2 * Iterations);
  EXPECT_EQ(CounterFree, 2 * Iterations);
  EXPECT_EQ(CounterAllocWithEvent, size_t{0});
  EXPECT_EQ(CounterFreeWithEvent, size_t{0});
  EXPECT_EQ(CounterEventsWait, size_t{0});
  EXPECT_EQ(CounterBarrier, size_t{0});
}

// The handler overloads on an out-of-order queue only request the event of the
// allocation if the submission returns one.
TEST_F(AsyncAllocTests, OutOfOrderHandlerOverloadNoAllocEvents) {
  queue Q = makeQueue(/*InOrder=*/false);
  auto Pool = Q.get_context().ext_oneapi_get_default_memory_pool(
      Q.get_device(), usm::alloc::device);

  void *Ptr = nullptr;
  void *PoolPtr = nullptr;
  oneapiext::submit(Q, [&](handler &CGH) {
    Ptr = oneapiext::async_malloc(CGH, usm::alloc::device, 1024);
  });
  oneapiext::submit(Q, [&](handler &CGH) {
    PoolPtr = oneapiext::async_malloc_from_pool(CGH, 1024, Pool);
  });
  Q.ext_oneapi_submit_barrier();
  oneapiext::submit(Q, [&](handler &CGH) { oneapiext::async_free(CGH, Ptr); });
  oneapiext::submit(Q,
                    [&](handler &CGH) { oneapiext::async_free(CGH, PoolPtr); });
  Q.wait();

  EXPECT_EQ(CounterAlloc, size_t{2});
  EXPECT_EQ(CounterAllocWithEvent, size_t{0});
  EXPECT_EQ(CounterFree, size_t{2});
  // The frees still request their event: on out-of-order queues the adapter
  // uses it to guard the reuse of the freed memory.
  EXPECT_EQ(CounterFreeWithEvent, size_t{2});
}

TEST_F(AsyncAllocTests, OutOfOrderHandlerOverloadEvents) {
  queue Q = makeQueue(/*InOrder=*/false);
  auto Pool = Q.get_context().ext_oneapi_get_default_memory_pool(
      Q.get_device(), usm::alloc::device);

  void *Ptr = nullptr;
  void *PoolPtr = nullptr;
  event AllocEvent = oneapiext::submit_with_event(Q, [&](handler &CGH) {
    Ptr = oneapiext::async_malloc(CGH, usm::alloc::device, 1024);
  });
  event PoolAllocEvent = oneapiext::submit_with_event(Q, [&](handler &CGH) {
    PoolPtr = oneapiext::async_malloc_from_pool(CGH, 1024, Pool);
  });
  oneapiext::submit(Q, [&](handler &CGH) {
    CGH.depends_on(AllocEvent);
    oneapiext::async_free(CGH, Ptr);
  });
  oneapiext::submit(Q, [&](handler &CGH) {
    CGH.depends_on(PoolAllocEvent);
    oneapiext::async_free(CGH, PoolPtr);
  });
  Q.wait();

  EXPECT_EQ(CounterAlloc, size_t{2});
  EXPECT_EQ(CounterAllocWithEvent, size_t{2});
  EXPECT_EQ(CounterFree, size_t{2});
  EXPECT_EQ(CounterFreeWithEvent, size_t{2});
}

// A host task dependency cannot be expressed to the backend, so the
// submission has to fall back to the scheduler. The allocation itself is still
// enqueued eagerly, as the pointer has to be returned to the caller
// immediately.
TEST_F(AsyncAllocTests, InOrderQueueShortcutAfterHostTask) {
  HostTaskExecuted = false;
  mock::getCallbacks().set_replace_callback(
      "urEnqueueUSMFreeExp", &redefined_urEnqueueUSMFreeExpAfterHostTask);

  queue Q = makeQueue(/*InOrder=*/true);

  std::mutex Mtx;
  std::unique_lock<std::mutex> Lock{Mtx};
  Q.submit([&](handler &CGH) {
    CGH.host_task([&]() {
      std::lock_guard<std::mutex> Guard{Mtx};
      HostTaskExecuted = true;
    });
  });

  void *Ptr = oneapiext::async_malloc(Q, usm::alloc::device, 1024);
  oneapiext::async_free(Q, Ptr);

  // The free must not have been enqueued to the backend yet, as it is ordered
  // after the host task, which is still blocked.
  EXPECT_EQ(CounterFree, size_t{0});

  Lock.unlock();
  Q.wait();

  EXPECT_EQ(CounterAlloc, size_t{1});
  EXPECT_EQ(CounterFree, size_t{1});
}

} // namespace
