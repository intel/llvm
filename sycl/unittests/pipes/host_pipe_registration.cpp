//==-------------- host_pipe_registration.cpp - Host pipe tests------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <cstring>

#include <gtest/gtest.h>
#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

namespace {
using namespace sycl;
using pipe_prop = decltype(ext::oneapi::experimental::properties(
    ext::intel::experimental::min_capacity<5>));

template <unsigned ID> struct pipe_id {
  static constexpr unsigned id = ID;
};

class test_data_type {
public:
  int num;
};

using test_host_pipe =
    ext::intel::experimental::host_pipe<pipe_id<0>, test_data_type, pipe_prop>;

pi_device_binary_struct generate_device_binary() {
  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data
  unittest::PiArray<unittest::PiOffloadEntry> Entries =
      unittest::makeEmptyKernels({"TestKernel"});
  unittest::PiPropertySet PropSet;
  pi_device_binary_struct MBinaryDesc = pi_device_binary_struct{
      PI_DEVICE_BINARY_VERSION,
      PI_DEVICE_BINARY_OFFLOAD_KIND_SYCL,
      PI_DEVICE_BINARY_TYPE_SPIRV,
      __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64,
      "",
      "",
      nullptr,
      nullptr,
      &*Bin.begin(),
      (&*Bin.begin()) + Bin.size(),
      Entries.begin(),
      Entries.end(),
      PropSet.begin(),
      PropSet.end(),
  };
  return MBinaryDesc;
}
pi_event READ = reinterpret_cast<pi_event>(0);
pi_event WRITE = reinterpret_cast<pi_event>(1);
static constexpr test_data_type PipeReadVal = {8};
static test_data_type PipeWriteVal = {0};
pi_result redefinedEnqueueReadHostPipe(pi_queue, pi_program, const char *,
                                       pi_bool, void *ptr, size_t, pi_uint32,
                                       const pi_event *, pi_event *event) {
  *(((test_data_type *)ptr)) = PipeReadVal;
  *event = READ;
  return PI_SUCCESS;
}
pi_result redefinedEnqueueWriteHostPipe(pi_queue, pi_program, const char *,
                                        pi_bool, void *ptr, size_t, pi_uint32,
                                        const pi_event *, pi_event *event) {
  test_data_type tmp = {9};
  PipeWriteVal = tmp;
  *event = WRITE;
  return PI_SUCCESS;
}

bool preparePiMock(platform &Plt) {
  if (Plt.is_host()) {
    std::cout << "Not run on host - no PI events created in that case"
              << std::endl;
    return false;
  }

  unittest::PiMock Mock{Plt};
  Mock.redefine<detail::PiApiKind::piextEnqueueReadHostPipe>(
      redefinedEnqueueReadHostPipe);
  Mock.redefine<detail::PiApiKind::piextEnqueueWriteHostPipe>(
      redefinedEnqueueWriteHostPipe);
  return true;
}

class PipeTest : public ::testing::Test {
protected:
  void SetUp() override {
    platform Plt{default_selector()};
    if (!preparePiMock(Plt))
      return;
    context Ctx{Plt.get_devices()[0]};
    queue Q{Ctx, default_selector()};
    plat = Plt;
    ctx = Ctx;
    q = Q;

    // Fake registration of host pipes
    sycl::detail::host_pipe_map::add(test_host_pipe::get_host_ptr(),
                                     "test_host_pipe_unique_id");
    // Fake registration of device image
    static constexpr size_t NumberOfImages = 1;
    pi_device_binary_struct MNativeImages[NumberOfImages];
    MNativeImages[0] = generate_device_binary();
    MAllBinaries = pi_device_binaries_struct{
        PI_DEVICE_BINARIES_VERSION,
        NumberOfImages,
        MNativeImages,
        nullptr, // not used, put here for compatibility with OpenMP
        nullptr, // not used, put here for compatibility with OpenMP
    };
    __sycl_register_lib(&MAllBinaries);
  }

  void TearDown() override { __sycl_unregister_lib(&MAllBinaries); }

  platform plat;
  context ctx;
  queue q;
  pi_device_binaries_struct MAllBinaries;
};

TEST_F(PipeTest, Basic) {
  const void *HostPipePtr = test_host_pipe::get_host_ptr();
  detail::HostPipeMapEntry *hostPipeEntry =
      detail::ProgramManager::getInstance().getHostPipeEntry(HostPipePtr);
  const std::string pipe_name = hostPipeEntry->MUniqueId;
  test_data_type host_pipe_read_data = {};
  void *data_ptr = &host_pipe_read_data;
  event e = q.submit([=](handler &CGH) {
    CGH.read_write_host_pipe(pipe_name, data_ptr, sizeof(test_data_type), false,
                             true /* read */);
  });
  e.wait();
  // auto host_pipe_read_data = test_host_pipe::read(q);
  assert(host_pipe_read_data.num == PipeReadVal.num);
  test_data_type tmp = {9};
  data_ptr = &tmp;
  event e_write = q.submit([=](handler &CGH) {
    CGH.read_write_host_pipe(pipe_name, data_ptr, sizeof(test_data_type), false,
                             false /* write */);
  });
  e_write.wait();
  // test_host_pipe::write(q, tmp);
  assert(PipeWriteVal.num == 9);
}

} // namespace
