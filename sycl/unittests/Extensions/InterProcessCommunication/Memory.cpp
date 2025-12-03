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

int DummyInt = 42;
void *DummyPtr = &DummyInt;

constexpr size_t DummyHandleDataSize = 10;
std::byte DummyHandleData[DummyHandleDataSize] = {
    std::byte{9}, std::byte{8}, std::byte{7}, std::byte{6}, std::byte{5},
    std::byte{4}, std::byte{3}, std::byte{2}, std::byte{1}, std::byte{0}};

thread_local int urIPCGetMemHandleExp_counter = 0;
thread_local int urIPCPutMemHandleExp_counter = 0;
thread_local int urIPCOpenMemHandleExp_counter = 0;
thread_local int urIPCCloseMemHandleExp_counter = 0;

ur_result_t replace_urIPCGetMemHandleExp(void *pParams) {
  ++urIPCGetMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_get_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.ppMem, DummyPtr);
  if (*params.ppIPCMemHandleDataSizeRet)
    **params.ppIPCMemHandleDataSizeRet = DummyHandleDataSize;
  if (*params.pppIPCMemHandleData)
    **params.pppIPCMemHandleData = DummyHandleData;
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCPutMemHandleExp(void *pParams) {
  ++urIPCPutMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_put_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.ppIPCMemHandleData, DummyHandleData);
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCOpenMemHandleExp(void *pParams) {
  ++urIPCOpenMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_open_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(
      memcmp(*params.ppIPCMemHandleData, DummyHandleData, DummyHandleDataSize),
      0);
  EXPECT_EQ(*params.pipcMemHandleDataSize, DummyHandleDataSize);
  **params.pppMem = DummyPtr;
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCCloseMemHandleExp(void *pParams) {
  ++urIPCCloseMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_close_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.ppMem, DummyPtr);
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_IPC_MEMORY_SUPPORT_EXP:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) = ur_bool_t{true};
    return UR_RESULT_SUCCESS;
  default:
    return UR_RESULT_SUCCESS;
  }
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

    mock::getCallbacks().set_replace_callback("urIPCGetMemHandleExp",
                                              replace_urIPCGetMemHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCPutMemHandleExp",
                                              replace_urIPCPutMemHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCOpenMemHandleExp",
                                              replace_urIPCOpenMemHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCCloseMemHandleExp",
                                              replace_urIPCCloseMemHandleExp);
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            after_urDeviceGetInfo);
  }

  sycl::unittest::UrMock<> Mock;
  sycl::context Ctxt;
};

TEST_F(IPCTests, IPCGetPutImplicit) {
  syclexp::ipc_memory::handle IPCMemHandle =
      syclexp::ipc_memory::get(DummyPtr, Ctxt);
  syclexp::ipc_memory::handle_data_t IPCMemHandleData = IPCMemHandle.data();
  EXPECT_EQ(IPCMemHandleData.size(), DummyHandleDataSize);
  EXPECT_EQ(IPCMemHandleData.data(), DummyHandleData);

  // Creating the IPC memory from a pointer should only call "get".
  EXPECT_EQ(urIPCGetMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCPutMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCOpenMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCCloseMemHandleExp_counter, 0);
}

TEST_F(IPCTests, IPCGetPutExplicit) {
  syclexp::ipc_memory::handle IPCMemHandle =
      syclexp::ipc_memory::get(DummyPtr, Ctxt);
  syclexp::ipc_memory::handle_data_t IPCMemHandleData = IPCMemHandle.data();
  EXPECT_EQ(IPCMemHandleData.size(), DummyHandleDataSize);
  EXPECT_EQ(IPCMemHandleData.data(), DummyHandleData);

  // Creating the IPC memory from a pointer should only call "get".
  EXPECT_EQ(urIPCGetMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCPutMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCOpenMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCCloseMemHandleExp_counter, 0);

  syclexp::ipc_memory::put(IPCMemHandle, Ctxt);

  // Calling "put" explicitly should call the UR function.
  EXPECT_EQ(urIPCGetMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCPutMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCOpenMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCCloseMemHandleExp_counter, 0);
}

TEST_F(IPCTests, IPCOpenClose) {
  syclexp::ipc_memory::handle_data_t HandleData{
      DummyHandleData, DummyHandleData + DummyHandleDataSize};
  void *Ptr =
      syclexp::ipc_memory::open(HandleData, Ctxt, Ctxt.get_devices()[0]);
  EXPECT_EQ(Ptr, DummyPtr);

  // Opening an IPC handle should call open.
  EXPECT_EQ(urIPCGetMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCPutMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCOpenMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCCloseMemHandleExp_counter, 0);

  syclexp::ipc_memory::close(Ptr, Ctxt);

  // When we close an IPC memory pointer, it should call close.
  EXPECT_EQ(urIPCGetMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCPutMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCOpenMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCCloseMemHandleExp_counter, 1);
}

TEST_F(IPCTests, IPCOpenCloseView) {
  syclexp::ipc_memory::handle_data_view_t HandleDataView{DummyHandleData,
                                                         DummyHandleDataSize};
  void *Ptr =
      syclexp::ipc_memory::open(HandleDataView, Ctxt, Ctxt.get_devices()[0]);
  EXPECT_EQ(Ptr, DummyPtr);

  // Opening an IPC handle should call open.
  EXPECT_EQ(urIPCGetMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCPutMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCOpenMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCCloseMemHandleExp_counter, 0);

  syclexp::ipc_memory::close(Ptr, Ctxt);

  // When we close an IPC memory pointer, it should call close.
  EXPECT_EQ(urIPCGetMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCPutMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCOpenMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCCloseMemHandleExp_counter, 1);
}

} // namespace
