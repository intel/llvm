//==------------------------------ IPC.cpp ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <helpers/UrMock.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

namespace {

int DummyInt1 = 42;
void *DummyPtr = &DummyInt1;

int DummyInt2 = 24;
ur_exp_ipc_mem_handle_t DummyMemHandle =
    reinterpret_cast<ur_exp_ipc_mem_handle_t>(&DummyInt2);

constexpr size_t DummyHandleDataSize = 10;
char DummyHandleData[DummyHandleDataSize] = {9, 8, 7, 6, 5, 4, 3, 2, 1};

thread_local int urIPCGetMemHandleExp_counter = 0;
thread_local int urIPCPutMemHandleExp_counter = 0;
thread_local int urIPCOpenMemHandleExp_counter = 0;
thread_local int urIPCCloseMemHandleExp_counter = 0;
thread_local int urIPCCreateMemHandleFromDataExp_counter = 0;
thread_local int urIPCDestroyMemHandleExp_counter = 0;
thread_local int urIPCGetMemHandleDataExp_counter = 0;

ur_result_t replace_urIPCGetMemHandleExp(void *pParams) {
  ++urIPCGetMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_get_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.ppMem, DummyPtr);
  **params.pphIPCMem = DummyMemHandle;
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCPutMemHandleExp(void *pParams) {
  ++urIPCPutMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_put_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.phIPCMem, DummyMemHandle);
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCOpenMemHandleExp(void *pParams) {
  ++urIPCOpenMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_open_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.phIPCMem, DummyMemHandle);
  **params.pppMem = DummyPtr;
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCCloseMemHandleExp(void *pParams) {
  ++urIPCCloseMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_close_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.ppMem, DummyPtr);
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCCreateMemHandleFromDataExp(void *pParams) {
  ++urIPCCreateMemHandleFromDataExp_counter;
  auto params =
      *static_cast<ur_ipc_create_mem_handle_from_data_exp_params_t *>(pParams);
  EXPECT_EQ(*params.pipcMemHandleData, DummyHandleData);
  EXPECT_EQ(*params.pipcMemHandleDataSize, DummyHandleDataSize);
  **params.pphIPCMem = DummyMemHandle;
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCDestroyMemHandleExp(void *pParams) {
  ++urIPCDestroyMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_destroy_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.phIPCMem, DummyMemHandle);
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCGetMemHandleDataExp(void *pParams) {
  ++urIPCGetMemHandleDataExp_counter;
  auto params =
      *static_cast<ur_ipc_get_mem_handle_data_exp_params_t *>(pParams);
  EXPECT_EQ(*params.phIPCMem, DummyMemHandle);
  **params.pppIPCHandleData = DummyHandleData;
  **params.ppIPCMemHandleDataSizeRet = DummyHandleDataSize;
  return UR_RESULT_SUCCESS;
}

class IPCTests : public ::testing::Test {
public:
  IPCTests() : Mock{}, Ctxt(sycl::platform()) {}

protected:
  void SetUp() override {
    urIPCGetMemHandleExp_counter = 0;
    urIPCPutMemHandleExp_counter = 0;
    urIPCOpenMemHandleExp_counter = 0;
    urIPCCloseMemHandleExp_counter = 0;
    urIPCCreateMemHandleFromDataExp_counter = 0;
    urIPCDestroyMemHandleExp_counter = 0;
    urIPCGetMemHandleDataExp_counter = 0;

    mock::getCallbacks().set_replace_callback("urIPCGetMemHandleExp",
                                              replace_urIPCGetMemHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCPutMemHandleExp",
                                              replace_urIPCPutMemHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCOpenMemHandleExp",
                                              replace_urIPCOpenMemHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCCloseMemHandleExp",
                                              replace_urIPCCloseMemHandleExp);
    mock::getCallbacks().set_replace_callback(
        "urIPCCreateMemHandleFromDataExp",
        replace_urIPCCreateMemHandleFromDataExp);
    mock::getCallbacks().set_replace_callback("urIPCDestroyMemHandleExp",
                                              replace_urIPCDestroyMemHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCGetMemHandleDataExp",
                                              replace_urIPCGetMemHandleDataExp);
  }

  sycl::unittest::UrMock<> Mock;
  sycl::context Ctxt;
};

TEST_F(IPCTests, IPCGetPut) {
  {
    syclexp::ipc_memory IPCMem{DummyPtr, Ctxt};

    // Creating the IPC memory from a pointer should only call "get".
    EXPECT_EQ(urIPCGetMemHandleExp_counter, 1);
    EXPECT_EQ(urIPCPutMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCOpenMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCCloseMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCCreateMemHandleFromDataExp_counter, 0);
    EXPECT_EQ(urIPCDestroyMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCGetMemHandleDataExp_counter, 0);

    sycl::span<const char, sycl::dynamic_extent> IPCMemHandleData =
        IPCMem.get_handle_data();
    EXPECT_EQ(IPCMemHandleData.data(), DummyHandleData);
    EXPECT_EQ(IPCMemHandleData.size(), DummyHandleDataSize);

    // Getting the underlying data should call the backend.
    EXPECT_EQ(urIPCGetMemHandleExp_counter, 1);
    EXPECT_EQ(urIPCPutMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCOpenMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCCloseMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCCreateMemHandleFromDataExp_counter, 0);
    EXPECT_EQ(urIPCDestroyMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCGetMemHandleDataExp_counter, 1);
  }

  // When the IPC memory object dies, it should return the handle, calling
  // "put".
  EXPECT_EQ(urIPCGetMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCPutMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCOpenMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCCloseMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCCreateMemHandleFromDataExp_counter, 0);
  EXPECT_EQ(urIPCDestroyMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCGetMemHandleDataExp_counter, 1);
}

TEST_F(IPCTests, IPCOpenClose) {
  {
    sycl::span<const char, sycl::dynamic_extent> HandleData{
        DummyHandleData, DummyHandleDataSize};
    syclexp::ipc_memory IPCMem{HandleData, Ctxt, Ctxt.get_devices()[0]};
    EXPECT_EQ(IPCMem.get_ptr(), DummyPtr);

    // Creating the IPC memory from handle data should first re-create the
    // handle and then call open on it.
    EXPECT_EQ(urIPCGetMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCPutMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCOpenMemHandleExp_counter, 1);
    EXPECT_EQ(urIPCCloseMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCCreateMemHandleFromDataExp_counter, 1);
    EXPECT_EQ(urIPCDestroyMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCGetMemHandleDataExp_counter, 0);

    sycl::span<const char, sycl::dynamic_extent> IPCMemHandleData =
        IPCMem.get_handle_data();
    EXPECT_EQ(IPCMemHandleData.data(), DummyHandleData);
    EXPECT_EQ(IPCMemHandleData.size(), DummyHandleDataSize);

    // Getting the underlying data should call the backend.
    EXPECT_EQ(urIPCGetMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCPutMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCOpenMemHandleExp_counter, 1);
    EXPECT_EQ(urIPCCloseMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCCreateMemHandleFromDataExp_counter, 1);
    EXPECT_EQ(urIPCDestroyMemHandleExp_counter, 0);
    EXPECT_EQ(urIPCGetMemHandleDataExp_counter, 1);
  }

  // When the IPC memory object dies, it should release the handle, calling
  // "close" and then destroying it.
  EXPECT_EQ(urIPCGetMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCPutMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCOpenMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCCloseMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCCreateMemHandleFromDataExp_counter, 1);
  EXPECT_EQ(urIPCDestroyMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCGetMemHandleDataExp_counter, 1);
}

} // namespace
