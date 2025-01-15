//==-------------- host_pipe_registration.cpp - Host pipe tests------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <sycl/sycl.hpp>

#include <detail/device_binary_image.hpp>
#include <detail/host_pipe_map_entry.hpp>
#include <gtest/gtest.h>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/detail/host_pipe_map.hpp>

class TestKernel;
MOCK_INTEGRATION_HEADER(TestKernel)

using namespace sycl;
using default_pipe_properties =
    decltype(sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::uses_valid<true>));

class PipeID;
using Pipe = sycl::ext::intel::experimental::pipe<PipeID, int, 10,
                                                  default_pipe_properties>;

static sycl::unittest::MockDeviceImage generateDefaultImage() {
  using namespace sycl::unittest;

  sycl::detail::host_pipe_map::add(Pipe::get_host_ptr(),
                                   "test_host_pipe_unique_id");

  MockPropertySet PropSet;
  MockProperty HostPipeInfo =
      makeHostPipeInfo("test_host_pipe_unique_id", sizeof(int));
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_HOST_PIPES,
                 std::vector<MockProperty>{std::move(HostPipeInfo)});

  std::vector<MockOffloadEntry> Entries = makeEmptyKernels({"TestKernel"});

  MockDeviceImage Img(std::move(Entries), std::move(PropSet));

  return Img;
}

ur_event_handle_t READ = reinterpret_cast<ur_event_handle_t>(0);
ur_event_handle_t WRITE = reinterpret_cast<ur_event_handle_t>(1);
static constexpr int PipeReadVal = 8;
static int PipeWriteVal = 0;
ur_result_t redefinedEnqueueReadHostPipe(void *pParams) {
  auto params = *static_cast<ur_enqueue_read_host_pipe_params_t *>(pParams);
  **params.pphEvent = mock::createDummyHandle<ur_event_handle_t>();
  *(((int *)(*params.ppDst))) = PipeReadVal;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEnqueueWriteHostPipe(void *pParams) {
  auto params = *static_cast<ur_enqueue_write_host_pipe_params_t *>(pParams);
  **params.pphEvent = mock::createDummyHandle<ur_event_handle_t>();
  PipeWriteVal = 9;
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  constexpr char MockSupportedExtensions[] =
      "cl_khr_fp64 cl_khr_fp16 cl_khr_il_program "
      "cl_intel_program_scope_host_pipe";
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_EXTENSIONS:
    if (*params.ppPropValue) {
      std::ignore = *params.ppropSize;
      assert(*params.ppropSize >= sizeof(MockSupportedExtensions));
      std::memcpy(*params.ppPropValue, MockSupportedExtensions,
                  sizeof(MockSupportedExtensions));
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(MockSupportedExtensions);
    return UR_RESULT_SUCCESS;
  default:;
  }
  return UR_RESULT_SUCCESS;
}

void prepareUrMock(unittest::UrMock<> &Mock) {
  mock::getCallbacks().set_replace_callback("urEnqueueReadHostPipe",
                                            &redefinedEnqueueReadHostPipe);
  mock::getCallbacks().set_replace_callback("urEnqueueWriteHostPipe",
                                            &redefinedEnqueueWriteHostPipe);
}

class PipeTest : public ::testing::Test {
public:
  PipeTest() : Mock{}, Plt{sycl::platform()} {}

protected:
  void SetUp() override {
    prepareUrMock(Mock);
    const sycl::device Dev = Plt.get_devices()[0];
    sycl::context Ctx{Dev};
    sycl::queue Q{Ctx, Dev};
    ctx = Ctx;
    q = Q;
  }

protected:
  unittest::UrMock<> Mock;
  sycl::platform Plt;
  context ctx;
  queue q;
};

static sycl::unittest::MockDeviceImage Img = generateDefaultImage();
static sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

TEST_F(PipeTest, Basic) {
  // Fake extension
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo);

  // Device registration

  // Testing read
  int HostPipeReadData;
  HostPipeReadData = Pipe::read(q);
  EXPECT_EQ(HostPipeReadData, PipeReadVal);

  // Testing write
  int HostPipeWriteData = 9;
  Pipe::write(q, HostPipeWriteData);
  EXPECT_EQ(PipeWriteVal, 9);
}

bool EventsWaitFails = true;
ur_result_t redefinedEventWait(void *) {
  return EventsWaitFails ? UR_RESULT_ERROR_UNKNOWN : UR_RESULT_SUCCESS;
}

ur_result_t after_urEventGetInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    if (*params.ppPropValue)
      *static_cast<ur_event_status_t *>(*params.ppPropValue) =
          ur_event_status_t(-1);
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_event_status_t);
  }
  return UR_RESULT_SUCCESS;
}

TEST_F(PipeTest, NonBlockingOperationFail) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo);
  mock::getCallbacks().set_replace_callback("urEventWait", &redefinedEventWait);

  bool Success = false;
  Pipe::read(q, Success);
  ASSERT_FALSE(Success);

  Pipe::write(q, 0, Success);
  ASSERT_FALSE(Success);

  // Test the OpenCL 1.0 case: no error code after waiting.
  EventsWaitFails = false;
  mock::getCallbacks().set_after_callback("urEventGetInfo",
                                          &after_urEventGetInfo);

  Pipe::read(q, Success);
  ASSERT_FALSE(Success);

  Pipe::write(q, 0, Success);
  ASSERT_FALSE(Success);
}
