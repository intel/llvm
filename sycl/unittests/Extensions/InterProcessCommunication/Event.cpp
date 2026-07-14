//==----------------------------- Event.cpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <detail/context_impl.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/oneapi/experimental/ipc_event.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace ipcevt = sycl::ext::oneapi::experimental::ipc::event;

namespace {

constexpr size_t DummyHandleDataSize = 12;
std::byte DummyHandleData[DummyHandleDataSize] = {
    std::byte{0}, std::byte{1}, std::byte{2},  std::byte{3},
    std::byte{4}, std::byte{5}, std::byte{6},  std::byte{7},
    std::byte{8}, std::byte{9}, std::byte{10}, std::byte{11}};

int DummyEventData = 0;
ur_event_handle_t DummyEvent = (ur_event_handle_t)&DummyEventData;

thread_local int urIPCGetEventHandleExp_counter = 0;
thread_local int urIPCPutEventHandleExp_counter = 0;
thread_local int urIPCOpenEventHandleExp_counter = 0;
thread_local int urEventRelease_counter = 0;

// Filled in by SetUp() from the live mock context — needed by the
// urEventGetInfo(UR_EVENT_INFO_CONTEXT) mock so that event_impl's constructor
// can validate that the imported event belongs to the right context.
ur_context_handle_t MockContextHandle = nullptr;

ur_result_t replace_urIPCGetEventHandleExp(void *pParams) {
  ++urIPCGetEventHandleExp_counter;
  auto params =
      *static_cast<ur_ipc_get_event_handle_exp_params_t *>(pParams);
  if (*params.pppIPCEventHandleData)
    **params.pppIPCEventHandleData = DummyHandleData;
  if (*params.ppIPCEventHandleDataSizeRet)
    **params.ppIPCEventHandleDataSizeRet = DummyHandleDataSize;
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCPutEventHandleExp(void *pParams) {
  ++urIPCPutEventHandleExp_counter;
  auto params =
      *static_cast<ur_ipc_put_event_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.ppIPCEventHandleData, (void *)DummyHandleData);
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCOpenEventHandleExp(void *pParams) {
  ++urIPCOpenEventHandleExp_counter;
  auto params =
      *static_cast<ur_ipc_open_event_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.pipcEventHandleDataSize, DummyHandleDataSize);
  EXPECT_EQ(
      memcmp(*params.ppIPCEventHandleData, DummyHandleData, DummyHandleDataSize),
      0);
  **params.pphEvent = DummyEvent;
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCOpenEventHandleExp_WrongSize(void *pParams) {
  ++urIPCOpenEventHandleExp_counter;
  return UR_RESULT_ERROR_INVALID_VALUE;
}

ur_result_t replace_urEventRelease(void *pParams) {
  ++urEventRelease_counter;
  return UR_RESULT_SUCCESS;
}

// event_impl(ur_event_handle_t, ctx) calls urEventGetInfo(UR_EVENT_INFO_CONTEXT)
// to validate the event belongs to the context. Return the mock context handle
// so the validation passes for imported events.
ur_result_t replace_urEventGetInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_EVENT_INFO_CONTEXT) {
    if (*params.ppPropValue)
      *static_cast<ur_context_handle_t *>(*params.ppPropValue) =
          MockContextHandle;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_context_handle_t);
  }
  return UR_RESULT_SUCCESS;
}

// Make the mock device advertise IPC event and reusable event support.
// Both are required: IPC events depend on reusable-event infrastructure.
ur_result_t after_urDeviceGetInfo_IPCEventSupport(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_IPC_EVENT_SUPPORT_EXP:
  case UR_DEVICE_INFO_REUSABLE_EVENTS_SUPPORT_EXP:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) = ur_bool_t{true};
    break;
  default:
    break;
  }
  return UR_RESULT_SUCCESS;
}

// Returns a dummy non-null event handle for urEventCreateExp, used by
// materializeIPCEvent (called from ipcevt::get on a not-yet-signaled event).
ur_result_t replace_urEventCreateExp(void *pParams) {
  auto params = *static_cast<ur_event_create_exp_params_t *>(pParams);
  if (*params.pphEvent)
    **params.pphEvent = DummyEvent;
  return UR_RESULT_SUCCESS;
}

class IPCEventTests : public ::testing::Test {
public:
  IPCEventTests() : Mock{}, Ctxt(sycl::platform()) {}

protected:
  void SetUp() override {
    urIPCGetEventHandleExp_counter = 0;
    urIPCPutEventHandleExp_counter = 0;
    urIPCOpenEventHandleExp_counter = 0;
    urEventRelease_counter = 0;

    MockContextHandle =
        sycl::detail::getSyclObjImpl(Ctxt)->getHandleRef();

    mock::getCallbacks().set_replace_callback("urIPCGetEventHandleExp",
                                              replace_urIPCGetEventHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCPutEventHandleExp",
                                              replace_urIPCPutEventHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCOpenEventHandleExp",
                                              replace_urIPCOpenEventHandleExp);
    mock::getCallbacks().set_replace_callback("urEventRelease",
                                              replace_urEventRelease);
    mock::getCallbacks().set_replace_callback("urEventGetInfo",
                                              replace_urEventGetInfo);
    mock::getCallbacks().set_replace_callback("urEventCreateExp",
                                              replace_urEventCreateExp);
    mock::getCallbacks().set_after_callback(
        "urDeviceGetInfo", after_urDeviceGetInfo_IPCEventSupport);
  }

  sycl::unittest::UrMock<> Mock;
  sycl::context Ctxt;
};

// make_event(enable_ipc) sets the IPC flag; the event is not yet backed by a
// UR handle (lazy materialization).
TEST_F(IPCEventTests, MakeEventIPCFlagSet) {
  sycl::event Evt =
      syclexp::make_event(Ctxt, syclexp::properties{syclexp::enable_ipc});
  EXPECT_TRUE(Evt.ext_oneapi_ipc_enabled());
}

// A default-constructed event is NOT IPC-enabled.
TEST_F(IPCEventTests, DefaultEventNotIPC) {
  sycl::event Evt;
  EXPECT_FALSE(Evt.ext_oneapi_ipc_enabled());
}

// make_event without enable_ipc produces an event where the flag is false.
TEST_F(IPCEventTests, MakeEventNoIPCFlag) {
  sycl::event Evt = syclexp::make_event(Ctxt, syclexp::properties{});
  EXPECT_FALSE(Evt.ext_oneapi_ipc_enabled());
}

// ipc::event::get calls urIPCGetEventHandleExp; the returned handle carries the
// correct data.
TEST_F(IPCEventTests, GetCallsURAndReturnsHandle) {
  sycl::event Evt =
      syclexp::make_event(Ctxt, syclexp::properties{syclexp::enable_ipc});

  sycl::ext::oneapi::experimental::ipc::handle H = ipcevt::get(Evt);

  EXPECT_EQ(urIPCGetEventHandleExp_counter, 1);
  EXPECT_EQ(urIPCPutEventHandleExp_counter, 0);
  EXPECT_EQ(urIPCOpenEventHandleExp_counter, 0);

  sycl::ext::oneapi::experimental::ipc::handle_data_t Data = H.data();
  ASSERT_EQ(Data.size(), DummyHandleDataSize);
  EXPECT_EQ(memcmp(Data.data(), DummyHandleData, DummyHandleDataSize), 0);
}

// ipc::event::get on a non-IPC event throws errc::invalid.
TEST_F(IPCEventTests, GetOnNonIPCEventThrows) {
  sycl::event Evt;
  bool caught = false;
  try {
    (void)ipcevt::get(Evt);
  } catch (const sycl::exception &E) {
    caught = true;
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid));
  }
  EXPECT_TRUE(caught);
  EXPECT_EQ(urIPCGetEventHandleExp_counter, 0);
}

// ipc::event::put calls urIPCPutEventHandleExp with the handle data.
TEST_F(IPCEventTests, PutCallsUR) {
  sycl::event Evt =
      syclexp::make_event(Ctxt, syclexp::properties{syclexp::enable_ipc});

  sycl::ext::oneapi::experimental::ipc::handle H = ipcevt::get(Evt);
  EXPECT_EQ(urIPCGetEventHandleExp_counter, 1);

  ipcevt::put(H, Ctxt);

  EXPECT_EQ(urIPCPutEventHandleExp_counter, 1);
  EXPECT_EQ(urIPCOpenEventHandleExp_counter, 0);
}

// ipc::event::open calls urIPCOpenEventHandleExp; the returned event is NOT
// IPC-enabled (imported events cannot be re-exported).
TEST_F(IPCEventTests, OpenCallsURAndImportedEventNotIPC) {
  sycl::ext::oneapi::experimental::ipc::handle_data_t HandleData{
      DummyHandleData, DummyHandleData + DummyHandleDataSize};

  sycl::event Imported = ipcevt::open(HandleData, Ctxt);

  EXPECT_EQ(urIPCGetEventHandleExp_counter, 0);
  EXPECT_EQ(urIPCPutEventHandleExp_counter, 0);
  EXPECT_EQ(urIPCOpenEventHandleExp_counter, 1);

  // Imported events cannot be re-exported.
  EXPECT_FALSE(Imported.ext_oneapi_ipc_enabled());
}

#if __cpp_lib_span
// Same test via the span (handle_data_view_t) overload.
TEST_F(IPCEventTests, OpenViewCallsUR) {
  sycl::ext::oneapi::experimental::ipc::handle_data_view_t HandleView{
      DummyHandleData, DummyHandleDataSize};

  sycl::event Imported = ipcevt::open(HandleView, Ctxt);

  EXPECT_EQ(urIPCOpenEventHandleExp_counter, 1);
  EXPECT_FALSE(Imported.ext_oneapi_ipc_enabled());
}
#endif

// ipc::event::open with a wrong-size buffer (UR returns
// UR_RESULT_ERROR_INVALID_VALUE) must throw errc::invalid.
TEST_F(IPCEventTests, OpenWrongSizeThrows) {
  mock::getCallbacks().set_replace_callback(
      "urIPCOpenEventHandleExp", replace_urIPCOpenEventHandleExp_WrongSize);

  // Any non-empty buffer will do; the mock rejects it unconditionally.
  sycl::ext::oneapi::experimental::ipc::handle_data_t Bogus(7, std::byte{0});
  bool caught = false;
  try {
    (void)ipcevt::open(Bogus, Ctxt);
  } catch (const sycl::exception &E) {
    caught = true;
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid));
  }
  EXPECT_TRUE(caught);
}

// make_event with enable_ipc on a context whose device lacks the aspect throws
// errc::feature_not_supported.
TEST_F(IPCEventTests, MakeEventNoAspectThrows) {
  // Remove the IPC-support callback so the device reports no support.
  mock::getCallbacks().set_after_callback("urDeviceGetInfo", nullptr);

  bool caught = false;
  try {
    (void)syclexp::make_event(Ctxt, syclexp::properties{syclexp::enable_ipc});
  } catch (const sycl::exception &E) {
    caught = true;
    EXPECT_EQ(E.code(),
              sycl::make_error_code(sycl::errc::feature_not_supported));
  }
  EXPECT_TRUE(caught);
}

// make_event with enable_ipc + enable_profiling simultaneously throws
// errc::invalid (they are mutually exclusive).
TEST_F(IPCEventTests, MakeEventIPCAndProfilingThrows) {
  bool caught = false;
  try {
    (void)syclexp::make_event(
        Ctxt, syclexp::properties{syclexp::enable_ipc, syclexp::enable_profiling});
  } catch (const sycl::exception &E) {
    caught = true;
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid));
  }
  EXPECT_TRUE(caught);
}

// ipc::event::open with a context whose device lacks the aspect throws
// errc::feature_not_supported.
TEST_F(IPCEventTests, OpenNoAspectThrows) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo", nullptr);

  sycl::ext::oneapi::experimental::ipc::handle_data_t HandleData{
      DummyHandleData, DummyHandleData + DummyHandleDataSize};
  bool caught = false;
  try {
    (void)ipcevt::open(HandleData, Ctxt);
  } catch (const sycl::exception &E) {
    caught = true;
    EXPECT_EQ(E.code(),
              sycl::make_error_code(sycl::errc::feature_not_supported));
  }
  EXPECT_TRUE(caught);
  EXPECT_EQ(urIPCOpenEventHandleExp_counter, 0);
}

} // namespace
