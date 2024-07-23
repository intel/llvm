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
#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>
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

static sycl::unittest::PiImage generateDefaultImage() {
  using namespace sycl::unittest;

  sycl::detail::host_pipe_map::add(Pipe::get_host_ptr(),
                                   "test_host_pipe_unique_id");

  PiPropertySet PropSet;
  PiProperty HostPipeInfo =
      makeHostPipeInfo("test_host_pipe_unique_id", sizeof(int));
  PropSet.insert(__SYCL_PI_PROPERTY_SET_SYCL_HOST_PIPES,
                 PiArray<PiProperty>{std::move(HostPipeInfo)});

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({"TestKernel"});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

pi_event READ = reinterpret_cast<pi_event>(0);
pi_event WRITE = reinterpret_cast<pi_event>(1);
static constexpr int PipeReadVal = 8;
static int PipeWriteVal = 0;
pi_result redefinedEnqueueReadHostPipe(pi_queue, pi_program, const char *,
                                       pi_bool, void *ptr, size_t, pi_uint32,
                                       const pi_event *, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  *(((int *)ptr)) = PipeReadVal;
  return PI_SUCCESS;
}
pi_result redefinedEnqueueWriteHostPipe(pi_queue, pi_program, const char *,
                                        pi_bool, void *ptr, size_t, pi_uint32,
                                        const pi_event *, pi_event *event) {
  *event = createDummyHandle<pi_event>();
  PipeWriteVal = 9;
  return PI_SUCCESS;
}

pi_result after_piDeviceGetInfo(pi_device device, pi_device_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  constexpr char MockSupportedExtensions[] =
      "cl_khr_fp64 cl_khr_fp16 cl_khr_il_program "
      "cl_intel_program_scope_host_pipe";
  switch (param_name) {
  case PI_DEVICE_INFO_EXTENSIONS: {
    if (param_value) {
      std::ignore = param_value_size;
      assert(param_value_size >= sizeof(MockSupportedExtensions));
      std::memcpy(param_value, MockSupportedExtensions,
                  sizeof(MockSupportedExtensions));
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(MockSupportedExtensions);
    return PI_SUCCESS;
  }
  default:;
  }
  return PI_SUCCESS;
}

void preparePiMock(unittest::PiMock &Mock) {
  Mock.redefine<detail::PiApiKind::piextEnqueueReadHostPipe>(
      redefinedEnqueueReadHostPipe);
  Mock.redefine<detail::PiApiKind::piextEnqueueWriteHostPipe>(
      redefinedEnqueueWriteHostPipe);
}

class PipeTest : public ::testing::Test {
public:
  PipeTest() : Mock{}, Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {
    preparePiMock(Mock);
    const sycl::device Dev = Plt.get_devices()[0];
    sycl::context Ctx{Dev};
    sycl::queue Q{Ctx, Dev};
    ctx = Ctx;
    q = Q;
  }

protected:
  unittest::PiMock Mock;
  sycl::platform Plt;
  context ctx;
  queue q;
};

static sycl::unittest::PiImage Img = generateDefaultImage();
static sycl::unittest::PiImageArray<1> ImgArray{&Img};

TEST_F(PipeTest, Basic) {
  // Fake extension
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo);

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
pi_result redefinedEventsWait(pi_uint32 num_events,
                              const pi_event *event_list) {
  return EventsWaitFails ? PI_ERROR_UNKNOWN : PI_SUCCESS;
}

pi_result after_piEventGetInfo(pi_event event, pi_event_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  if (param_name == PI_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    if (param_value)
      *static_cast<pi_event_status *>(param_value) = pi_event_status(-1);
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_event_status);
  }
  return PI_SUCCESS;
}

TEST_F(PipeTest, NonBlockingOperationFail) {
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo);
  Mock.redefine<sycl::detail::PiApiKind::piEventsWait>(redefinedEventsWait);

  bool Success = false;
  Pipe::read(q, Success);
  ASSERT_FALSE(Success);

  Pipe::write(q, 0, Success);
  ASSERT_FALSE(Success);

  // Test the OpenCL 1.0 case: no error code after waiting.
  EventsWaitFails = false;
  Mock.redefineAfter<sycl::detail::PiApiKind::piEventGetInfo>(
      after_piEventGetInfo);

  Pipe::read(q, Success);
  ASSERT_FALSE(Success);

  Pipe::write(q, 0, Success);
  ASSERT_FALSE(Success);
}
