//==------------------------- PhysicalMemory.cpp ----------------------------==//
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
#include <sycl/ext/oneapi/experimental/ipc_physical_memory.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

namespace {

constexpr size_t PhysMemSize = 8;
int DummyPhysicalMemHandleData = 123;
ur_physical_mem_handle_t DummyPhysicalMemHandle =
  (ur_physical_mem_handle_t)&DummyPhysicalMemHandleData;
constexpr size_t DummyHandleDataSize = 10;
std::byte DummyHandleData[DummyHandleDataSize] = {
    std::byte{9}, std::byte{8}, std::byte{7}, std::byte{6}, std::byte{5},
    std::byte{4}, std::byte{3}, std::byte{2}, std::byte{1}, std::byte{0}};


int urPhysicalMemCreate_counter = 0;
int urPhysicalMemRelease_counter = 0;
int urPhysicalMemGetInfo_counter = 0;
int urIPCGetPhysMemHandleExp_counter = 0;
int urIPCPutPhysMemHandleExp_counter = 0;
int urIPCOpenPhysMemHandleExp_counter = 0;
int urIPCClosePhysMemHandleExp_counter = 0;


ur_result_t replace_urPhysicalMemCreate(void *pParams) {
  ++urPhysicalMemCreate_counter;
  auto params = *static_cast<ur_physical_mem_create_params_t *>(pParams);
  EXPECT_EQ(*params.psize, PhysMemSize);
  EXPECT_TRUE(*params.ppProperties != nullptr);
  EXPECT_TRUE((*params.ppProperties)->flags & UR_PHYSICAL_MEM_FLAG_ENABLE_IPC);
  **params.pphPhysicalMem = DummyPhysicalMemHandle;
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urPhysicalMemRelease(void *pParams) {
  ++urPhysicalMemRelease_counter;
  auto params = *static_cast<ur_physical_mem_release_params_t *>(pParams);
  EXPECT_EQ(*params.phPhysicalMem, DummyPhysicalMemHandle);
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urPhysicalMemGetInfo(void *pParams) {
  ++urPhysicalMemGetInfo_counter;
  auto params = *static_cast<ur_physical_mem_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_PHYSICAL_MEM_INFO_SIZE:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(size_t);
    if (*params.ppPropValue)
      *static_cast<size_t *>(*params.ppPropValue) = PhysMemSize;
    return UR_RESULT_SUCCESS;
  default:
    return UR_RESULT_SUCCESS;
  }
}

ur_result_t replace_urIPCGetPhysMemHandleExp(void *pParams) {
  ++urIPCGetPhysMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_get_phys_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.phPhysMem, DummyPhysicalMemHandle);
  if (*params.ppIPCPhysMemHandleDataSizeRet)
    **params.ppIPCPhysMemHandleDataSizeRet = DummyHandleDataSize;
  if (*params.pppIPCPhysMemHandleData)
    **params.pppIPCPhysMemHandleData = DummyHandleData;
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCPutPhysMemHandleExp(void *pParams) {
  ++urIPCPutPhysMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_put_phys_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.ppIPCPhysMemHandleData, (void *)DummyHandleData);
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCOpenPhysMemHandleExp(void *pParams) {
  ++urIPCOpenPhysMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_open_phys_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(
      memcmp(*params.ppIPCPhysMemHandleData, DummyHandleData, DummyHandleDataSize),
      0);
  EXPECT_EQ(*params.pipcPhysMemHandleDataSize, DummyHandleDataSize);
  **params.pphPhysMem = DummyPhysicalMemHandle;
  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urIPCClosePhysMemHandleExp(void *pParams) {
  ++urIPCClosePhysMemHandleExp_counter;
  auto params = *static_cast<ur_ipc_close_phys_mem_handle_exp_params_t *>(pParams);
  EXPECT_EQ(*params.phPhysMem, DummyPhysicalMemHandle);
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_IPC_PHYSICAL_MEMORY_SUPPORT_EXP:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) = ur_bool_t{true};
    return UR_RESULT_SUCCESS;
  default:
    return UR_RESULT_SUCCESS;
  }
}

class IPCPhysMemTests : public ::testing::Test {
public:
  IPCPhysMemTests() : Mock{}, Ctxt(sycl::platform()),
    Dev(sycl::platform().get_devices()[0]) {}

protected:
  void SetUp() override {
    urPhysicalMemCreate_counter = 0;
    urPhysicalMemRelease_counter = 0;
    urPhysicalMemGetInfo_counter = 0;
    urIPCGetPhysMemHandleExp_counter = 0;
    urIPCPutPhysMemHandleExp_counter = 0;
    urIPCOpenPhysMemHandleExp_counter = 0;
    urIPCClosePhysMemHandleExp_counter = 0;

    mock::getCallbacks().set_replace_callback("urPhysicalMemCreate",
                                              replace_urPhysicalMemCreate);
    mock::getCallbacks().set_replace_callback("urPhysicalMemRelease",
                                              replace_urPhysicalMemRelease);
    mock::getCallbacks().set_replace_callback("urPhysicalMemGetInfo",
                                              replace_urPhysicalMemGetInfo);
    mock::getCallbacks().set_replace_callback("urIPCGetPhysMemHandleExp",
                                              replace_urIPCGetPhysMemHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCPutPhysMemHandleExp",
                                              replace_urIPCPutPhysMemHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCOpenPhysMemHandleExp",
                                              replace_urIPCOpenPhysMemHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCClosePhysMemHandleExp",
                                              replace_urIPCClosePhysMemHandleExp);
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            after_urDeviceGetInfo);
  }

  sycl::unittest::UrMock<> Mock;
  sycl::context Ctxt;
  sycl::device Dev;
};

TEST_F(IPCPhysMemTests, IPCGetPut) {
  {
    syclexp::properties PropList{syclexp::enable_ipc};
    syclexp::physical_mem PhysMem{Dev, Ctxt, PhysMemSize, PropList};

    syclexp::ipc::handle IPCMemHandle =
        syclexp::ipc::physical_memory::get(PhysMem);
    syclexp::ipc::handle_data_t IPCMemHandleData = IPCMemHandle.data();
    ASSERT_EQ(IPCMemHandleData.size(), DummyHandleDataSize);

    EXPECT_EQ(std::memcmp(IPCMemHandleData.data(), DummyHandleData,
                          DummyHandleDataSize),
              0);

    syclexp::ipc::physical_memory::put(IPCMemHandle);
  }

  EXPECT_EQ(urPhysicalMemCreate_counter, 1);
  EXPECT_EQ(urPhysicalMemRelease_counter, 1);
  EXPECT_EQ(urPhysicalMemGetInfo_counter, 0);
  EXPECT_EQ(urIPCGetPhysMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCPutPhysMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCOpenPhysMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCClosePhysMemHandleExp_counter, 0);
}

TEST_F(IPCPhysMemTests, IPCOpenClose) {
  {
    syclexp::ipc::handle_data_t HandleData{
        DummyHandleData, DummyHandleData + DummyHandleDataSize};
    syclexp::physical_mem PhysMem =
        syclexp::ipc::physical_memory::open(HandleData);
    EXPECT_EQ(PhysMem.size(), PhysMemSize);
  }
  EXPECT_EQ(urPhysicalMemCreate_counter, 0);
  EXPECT_EQ(urPhysicalMemRelease_counter, 0);
  EXPECT_EQ(urPhysicalMemGetInfo_counter, 1);
  EXPECT_EQ(urIPCGetPhysMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCPutPhysMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCOpenPhysMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCClosePhysMemHandleExp_counter, 1);
}

TEST_F(IPCPhysMemTests, IPCOpenCloseView) {
  {
    syclexp::ipc::handle_data_view_t HandleDataView{DummyHandleData,
                                                    DummyHandleDataSize};
    syclexp::physical_mem PhysMem =
        syclexp::ipc::physical_memory::open(HandleDataView);
    EXPECT_EQ(PhysMem.size(), PhysMemSize);
  }
  EXPECT_EQ(urPhysicalMemCreate_counter, 0);
  EXPECT_EQ(urPhysicalMemRelease_counter, 0);
  EXPECT_EQ(urPhysicalMemGetInfo_counter, 1);
  EXPECT_EQ(urIPCGetPhysMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCPutPhysMemHandleExp_counter, 0);
  EXPECT_EQ(urIPCOpenPhysMemHandleExp_counter, 1);
  EXPECT_EQ(urIPCClosePhysMemHandleExp_counter, 1);
}

} // namespace
