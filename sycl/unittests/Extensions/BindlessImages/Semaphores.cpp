#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <detail/event_impl.hpp>
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
thread_local ur_event_handle_t
    urBindlessImagesWaitExternalSemaphoreExp_lastEvent = nullptr;
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
  EXPECT_EQ(*Params.pnumEventsInWaitList, uint32_t{0});
  EXPECT_EQ(*Params.pphEventWaitList, nullptr);
  EXPECT_NE(*Params.pphEvent, nullptr);
  if (*Params.pphEvent) {
    urBindlessImagesWaitExternalSemaphoreExp_lastEvent =
        mock::createDummyHandle<ur_event_handle_t>();
    **Params.pphEvent = urBindlessImagesWaitExternalSemaphoreExp_lastEvent;
  }
  return UR_RESULT_SUCCESS;
}

thread_local int urBindlessImagesSignalExternalSemaphoreExp_counter = 0;
thread_local bool
    urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = false;
thread_local uint32_t
    urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 0;
thread_local ur_event_handle_t
    urBindlessImagesSignalExternalSemaphoreExp_lastEvent = nullptr;
inline ur_result_t
urBindlessImagesSignalExternalSemaphoreExp_replace(void *pParams) {
  ++urBindlessImagesSignalExternalSemaphoreExp_counter;
  ur_bindless_images_signal_external_semaphore_exp_params_t Params =
      *reinterpret_cast<
          ur_bindless_images_signal_external_semaphore_exp_params_t *>(pParams);
  EXPECT_EQ(*Params.phasSignalValue,
            urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue);
  if (urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue) {
    EXPECT_EQ(*Params.psignalValue, SignalValue);
  }
  EXPECT_EQ(*Params.pnumEventsInWaitList,
            urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents);
  if (urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents) {
    EXPECT_NE(*Params.pphEventWaitList, nullptr);
  } else {
    EXPECT_EQ(*Params.pphEventWaitList, nullptr);
  }
  EXPECT_NE(*Params.pphEvent, nullptr);
  if (*Params.pphEvent) {
    urBindlessImagesSignalExternalSemaphoreExp_lastEvent =
        mock::createDummyHandle<ur_event_handle_t>();
    **Params.pphEvent = urBindlessImagesSignalExternalSemaphoreExp_lastEvent;
  }
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

  DummySemaphore.handle_type =
      syclexp::external_semaphore_handle_type::opaque_fd;

  urBindlessImagesWaitExternalSemaphoreExp_expectHasWaitValue = false;
  sycl::event E = Q.ext_oneapi_wait_external_semaphore(DummySemaphore);
  EXPECT_EQ(urBindlessImagesWaitExternalSemaphoreExp_counter, 1);
  EXPECT_EQ(sycl::detail::getSyclObjImpl(E)->getHandle(),
            urBindlessImagesWaitExternalSemaphoreExp_lastEvent);

  DummySemaphore.handle_type =
      syclexp::external_semaphore_handle_type::timeline_fd;

  urBindlessImagesWaitExternalSemaphoreExp_expectHasWaitValue = true;
  E = Q.ext_oneapi_wait_external_semaphore(DummySemaphore, WaitValue);
  EXPECT_EQ(urBindlessImagesWaitExternalSemaphoreExp_counter, 2);
  EXPECT_EQ(sycl::detail::getSyclObjImpl(E)->getHandle(),
            urBindlessImagesWaitExternalSemaphoreExp_lastEvent);
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
  int DummyInt1 = 0, DummyInt2 = 0;
  syclexp::external_semaphore DummySemaphore{};
  DummySemaphore.raw_handle =
      reinterpret_cast<ur_exp_external_semaphore_handle_t>(&DummyInt1);

  // We create dummy events with dummy UR handles to make the runtime think we
  // pass actual device events.
  auto DummyEventImpl1 = sycl::detail::event_impl::create_device_event(
      *sycl::detail::getSyclObjImpl(Q));
  auto DummyEventImpl2 = sycl::detail::event_impl::create_device_event(
      *sycl::detail::getSyclObjImpl(Q));
  DummyEventImpl1->setHandle(reinterpret_cast<ur_event_handle_t>(&DummyInt1));
  DummyEventImpl2->setHandle(reinterpret_cast<ur_event_handle_t>(&DummyInt2));
  sycl::event DummyEvent1 =
      sycl::detail::createSyclObjFromImpl<sycl::event>(DummyEventImpl1);
  sycl::event DummyEvent2 =
      sycl::detail::createSyclObjFromImpl<sycl::event>(DummyEventImpl2);
  std::vector<sycl::event> DummyEventList{DummyEvent1, DummyEvent2};

  DummySemaphore.handle_type =
      syclexp::external_semaphore_handle_type::opaque_fd;

  urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = false;
  urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 0;
  sycl::event E = Q.ext_oneapi_signal_external_semaphore(DummySemaphore);
  EXPECT_EQ(urBindlessImagesSignalExternalSemaphoreExp_counter, 1);
  EXPECT_EQ(sycl::detail::getSyclObjImpl(E)->getHandle(),
            urBindlessImagesSignalExternalSemaphoreExp_lastEvent);

  urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = false;
  urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 1;
  E = Q.ext_oneapi_signal_external_semaphore(DummySemaphore, DummyEvent1);
  EXPECT_EQ(urBindlessImagesSignalExternalSemaphoreExp_counter, 2);
  EXPECT_EQ(sycl::detail::getSyclObjImpl(E)->getHandle(),
            urBindlessImagesSignalExternalSemaphoreExp_lastEvent);

  urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = false;
  urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 2;
  E = Q.ext_oneapi_signal_external_semaphore(DummySemaphore, DummyEventList);
  EXPECT_EQ(urBindlessImagesSignalExternalSemaphoreExp_counter, 3);
  EXPECT_EQ(sycl::detail::getSyclObjImpl(E)->getHandle(),
            urBindlessImagesSignalExternalSemaphoreExp_lastEvent);

  DummySemaphore.handle_type =
      syclexp::external_semaphore_handle_type::timeline_fd;

  urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = true;
  urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 0;
  E = Q.ext_oneapi_signal_external_semaphore(DummySemaphore, SignalValue);
  EXPECT_EQ(urBindlessImagesSignalExternalSemaphoreExp_counter, 4);
  EXPECT_EQ(sycl::detail::getSyclObjImpl(E)->getHandle(),
            urBindlessImagesSignalExternalSemaphoreExp_lastEvent);

  urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = true;
  urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 1;
  E = Q.ext_oneapi_signal_external_semaphore(DummySemaphore, SignalValue,
                                             DummyEvent1);
  EXPECT_EQ(urBindlessImagesSignalExternalSemaphoreExp_counter, 5);
  EXPECT_EQ(sycl::detail::getSyclObjImpl(E)->getHandle(),
            urBindlessImagesSignalExternalSemaphoreExp_lastEvent);

  urBindlessImagesSignalExternalSemaphoreExp_expectHasSignalValue = true;
  urBindlessImagesSignalExternalSemaphoreExp_expectedNumWaitEvents = 2;
  E = Q.ext_oneapi_signal_external_semaphore(DummySemaphore, SignalValue,
                                             DummyEventList);
  EXPECT_EQ(urBindlessImagesSignalExternalSemaphoreExp_counter, 6);
  EXPECT_EQ(sycl::detail::getSyclObjImpl(E)->getHandle(),
            urBindlessImagesSignalExternalSemaphoreExp_lastEvent);
}
