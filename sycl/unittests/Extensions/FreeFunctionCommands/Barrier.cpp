#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/khr/free_function_commands.hpp>

thread_local size_t NumOfEnqueueEventsWaitWithBarrier = 0;

inline ur_result_t after_urEnqueueEventsWaitWithBarrierExt(void *pParams) {
  (void)pParams;
  ++NumOfEnqueueEventsWaitWithBarrier;
  return UR_RESULT_SUCCESS;
}

TEST(BarrierTest, CommandBarrierShortcut) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &after_urEnqueueEventsWaitWithBarrierExt);

  NumOfEnqueueEventsWaitWithBarrier = 0;
  sycl::queue Queue;
  sycl::khr::command_barrier(Queue);
  ASSERT_EQ(NumOfEnqueueEventsWaitWithBarrier, size_t{1});
}

TEST(BarrierTest, CommandBarrier) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &after_urEnqueueEventsWaitWithBarrierExt);

  NumOfEnqueueEventsWaitWithBarrier = 0;

  sycl::queue Queue;

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::command_barrier(Handler);
  });

  ASSERT_EQ(NumOfEnqueueEventsWaitWithBarrier, size_t{1});
}

TEST(BarrierTest, EventBarrierShortcut) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &after_urEnqueueEventsWaitWithBarrierExt);

  NumOfEnqueueEventsWaitWithBarrier = 0;

  sycl::queue Queue1;
  sycl::queue Queue2;

  sycl::event Event1 = Queue1.single_task<TestKernel<>>([]() {});
  sycl::event Event2 = Queue2.single_task<TestKernel<>>([]() {});

  sycl::khr::event_barrier(Queue2, {Event1, Event2});

  ASSERT_EQ(NumOfEnqueueEventsWaitWithBarrier, size_t{1});
}

TEST(BarrierTest, EventBarrier) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &after_urEnqueueEventsWaitWithBarrierExt);

  NumOfEnqueueEventsWaitWithBarrier = 0;

  sycl::queue Queue1;
  sycl::queue Queue2;

  sycl::event Event1 = Queue1.single_task<TestKernel<>>([]() {});
  sycl::event Event2 = Queue2.single_task<TestKernel<>>([]() {});

  sycl::khr::submit(Queue2, [&](sycl::handler &Handler) {
    sycl::khr::event_barrier(Handler, {Event1, Event2});
  });

  ASSERT_EQ(NumOfEnqueueEventsWaitWithBarrier, size_t{1});
}
