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
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>
#include <sycl/detail/host_pipe_map.hpp>

template <size_t KernelSize = 1> class TestKernel;

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
template <size_t KernelSize> struct KernelInfo<TestKernel<KernelSize>> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "TestKernel"; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() { return KernelSize; }
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

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
  *(((int *)ptr)) = PipeReadVal;
  return PI_SUCCESS;
}
pi_result redefinedEnqueueWriteHostPipe(pi_queue, pi_program, const char *,
                                        pi_bool, void *ptr, size_t, pi_uint32,
                                        const pi_event *, pi_event *event) {
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
      assert(param_value_size >= sizeof(MockSupportedExtensions));
      std::memcpy(param_value, MockSupportedExtensions,
                  sizeof(MockSupportedExtensions));
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(MockSupportedExtensions);
    return PI_SUCCESS;
  }
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

TEST_F(PipeTest, Basic) {
  // Fake extension
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo);

  // Device registration
  static sycl::unittest::PiImage Img = generateDefaultImage();
  static sycl::unittest::PiImageArray<1> ImgArray{&Img};

  // Testing read
  int HostPipeReadData;
  HostPipeReadData = Pipe::read(q);
  assert(HostPipeReadData == PipeReadVal);

  // Testing write
  int HostPipeWriteData = 9;
  Pipe::write(q, HostPipeWriteData);
  assert(PipeWriteVal == 9);
}
