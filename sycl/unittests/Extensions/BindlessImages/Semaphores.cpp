#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <sycl/queue.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

constexpr uint64_t WaitValue = 42;
constexpr uint64_t SignalValue = 24;

thread_local int urBindlessImagesWaitExternalSemaphoreExp_counter = 0;
thread_local bool urBindlessImagesWaitExternalSemaphoreExp_expectHasWaitValue =
    false;
inline ur_result_t
urBindlessImagesWaitExternalSemaphoreExp_replace(void *pParams) {
  ++urBindlessImagesWaitExternalSemaphoreExp_counter;
  ur_bindless_images_wait_external_semaphore_exp_params_t Params =
      *reinterpret_cast<
          ur_bindless_images_wait_external_semaphore_exp_params_t *>(pParams);
  EXPECT_EQ(*Params.phasWaitValue,
            urBindlessImagesWaitExternalSemaphoreExp_expectHasWaitValue);
  if (urBindlessImagesWaitExternalSemaphoreExp_expectHasWaitValue) {
    EXPECT_EQ(*Params.pwaitValue, WaitValue);
  }
  EXPECT_EQ(*Params.pphEvent, nullptr);
  EXPECT_EQ(*Params.pnumEventsInWaitList, uint32_t{0});
  EXPECT_NE(*Params.pphEventWaitList, nullptr);
  return UR_RESULT_SUCCESS;
}

thread_local int urBindlessImagesSignalExternalSemaphoreExp_counter = 0;
thread_local bool
    urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = false;
thread_local uint32_t
    urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 0;
inline ur_result_t
urBindlessImagesSignalExternalSemaphoreExp_replace(void *pParams) {
  ++urBindlessImagesSignalExternalSemaphoreExp_counter;
  ur_bindless_images_signal_external_semaphore_exp_params_t Params =
      *reinterpret_cast<
          ur_bindless_images_signal_external_semaphore_exp_params_t *>(pParams);
  EXPECT_EQ(*Params.pphEvent, nullptr);
  EXPECT_EQ(*Params.phasSignalValue,
            urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue);
  if (urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue) {
    EXPECT_EQ(*Params.psignalValue, SignalValue);
  }
  if (urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents) {
    EXPECT_NE(*Params.pphEvent, nullptr);
  }

  else {
    EXPECT_EQ(*Params.pphEvent, nullptr);
  }
  EXPECT_EQ(*Params.pnumEventsInWaitList,
            urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents);
  EXPECT_NE(*Params.pphEventWaitList, nullptr);
  return UR_RESULT_SUCCESS;
}

TEST(BindlessImagesExtensionTests, ExternalSemaphoreWait) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback(
      "urBindlessImagesWaitExternalSemaphoreExp",
      &urBindlessImagesWaitExternalSemaphoreExp_replace);
  urBindlessImagesWaitExternalSemaphoreExp_counter = 0;

  sycl::queue Q;

  // Create a dummy external semaphore and set the raw handle to some dummy.
  // The mock implementation should never access the handle, so this is safe.
  int DummyInt = 0;
  syclexp::external_semaphore DummySemaphore{};
  DummySemaphore.raw_handle =
      reinterpret_cast<ur_exp_external_semaphore_handle_t>(&DummyInt);

  urBindlessImagesWaitExternalSemaphoreExp_expectHasWaitValue = false;
  Q.ext_oneapi_wait_external_semaphore(DummySemaphore);
  EXPECT_EQ(urBindlessImagesWaitExternalSemaphoreExp_counter, 1);

  urBindlessImagesWaitExternalSemaphoreExp_expectHasWaitValue = true;
  Q.ext_oneapi_wait_external_semaphore(DummySemaphore, WaitValue);
  EXPECT_EQ(urBindlessImagesWaitExternalSemaphoreExp_counter, 2);
}

TEST(BindlessImagesExtensionTests, ExternalSemaphoreSignal) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback(
      "urBindlessImagesSignalExternalSemaphoreExp",
      &urBindlessImagesSignalExternalSemaphoreExp_replace);
  urBindlessImagesSignalExternalSemaphoreExp_counter = 0;

  sycl::queue Q;

  // Create a dummy external semaphore and set the raw handle to some dummy.
  // The mock implementation should never access the handle, so this is safe.
  int DummyInt = 0;
  syclexp::external_semaphore DummySemaphore{};
  DummySemaphore.raw_handle =
      reinterpret_cast<ur_exp_external_semaphore_handle_t>(&DummyInt);

  urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = false;
  urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 0;
  Q.ext_oneapi_signal_external_semaphore(DummySemaphore);
  EXPECT_EQ(urBindlessImagesSignalExternalSemaphoreExp_counter, 1);

  urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = true;
  urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 0;
  Q.ext_oneapi_signal_external_semaphore(DummySemaphore, SignalValue);
  EXPECT_EQ(urBindlessImagesSignalExternalSemaphoreExp_counter, 2);

  urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = false;
  urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 1;
  Q.ext_oneapi_signal_external_semaphore(DummySemaphore, sycl::event{});
  EXPECT_EQ(urBindlessImagesSignalExternalSemaphoreExp_counter, 3);

  urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = true;
  urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 1;
  Q.ext_oneapi_signal_external_semaphore(DummySemaphore, SignalValue,
                                         sycl::event{});
  EXPECT_EQ(urBindlessImagesSignalExternalSemaphoreExp_counter, 4);

  std::vector<sycl::event> DummyEventList(2);

  urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = false;
  urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 2;
  Q.ext_oneapi_signal_external_semaphore(DummySemaphore, DummyEventList);
  EXPECT_EQ(urBindlessImagesSignalExternalSemaphoreExp_counter, 5);

  urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = true;
  urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 2;
  Q.ext_oneapi_signal_external_semaphore(DummySemaphore, SignalValue,
                                         DummyEventList);
  EXPECT_EQ(urBindlessImagesSignalExternalSemaphoreExp_counter, 6);
}
