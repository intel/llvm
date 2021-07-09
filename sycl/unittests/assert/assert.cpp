//==---------- assert.cpp --- Check assert helpers enqueue -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <set>
#include <thread>

class TestKernel;

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <> struct KernelInfo<TestKernel> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "TestKernel"; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

static constexpr const kernel_param_desc_t Signatures[] = {
    {kernel_param_kind_t::kind_accessor, 4062, 0}};

template <> struct KernelInfo<::sycl::detail::AssertInfoCopier> {
  static constexpr const char *getName() {
    return "_ZTSN2cl4sycl6detail16AssertInfoCopierE";
  }
  static constexpr unsigned getNumParams() { return 1; }
  static constexpr const kernel_param_desc_t &getParamDesc(unsigned Idx) {
    assert(!Idx);
    return Signatures[Idx];
  }
  static constexpr bool isESIMD() { return 0; }
  static constexpr bool callsThisItem() { return 0; }
  static constexpr bool callsAnyThisFreeFunction() { return 0; }
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

static sycl::unittest::PiImage generateDefaultImage() {
  using namespace sycl::unittest;

  static const std::string KernelName = "TestKernel";
  static const std::string CopierKernelName =
      "_ZTSN2cl4sycl6detail16AssertInfoCopierE";

  PiPropertySet PropSet;

  setKernelUsesAssert({KernelName}, PropSet);

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({KernelName});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

static sycl::unittest::PiImage generateCopierKernelImage() {
  using namespace sycl::unittest;

  static const std::string CopierKernelName =
      "_ZTSN2cl4sycl6detail16AssertInfoCopierE";

  PiPropertySet PropSet;

  std::vector<unsigned char> Bin{10, 11, 12, 13, 14, 15}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({CopierKernelName});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

sycl::unittest::PiImage Imgs[] = {generateDefaultImage(),
                                  generateCopierKernelImage()};
sycl::unittest::PiImageArray<2> ImgArray{Imgs};

static int KernelLaunchCounter = 0;
static std::mutex WaitedEventsMutex;
static std::set<int> WaitedEvents;
static constexpr int PauseWaitOnIdx = 1;
static std::atomic<bool> StartedWait{false};
static std::atomic<bool> ContinueWait{false};
static std::atomic<bool> PausedWaitDone{false};

// Mock redifinitions
static pi_result redefinedProgramCreate(pi_context, const void *, size_t,
                                        pi_program *) {
  return PI_SUCCESS;
}

static pi_result redefinedProgramGetInfo(pi_program program,
                                         pi_program_info param_name,
                                         size_t param_value_size,
                                         void *param_value,
                                         size_t *param_value_size_ret) {
  if (param_name == PI_PROGRAM_INFO_NUM_DEVICES) {
    auto value = reinterpret_cast<unsigned int *>(param_value);
    *value = 1;
  }

  if (param_name == PI_PROGRAM_INFO_BINARY_SIZES) {
    auto value = reinterpret_cast<size_t *>(param_value);
    value[0] = 1;
  }

  if (param_name == PI_PROGRAM_INFO_BINARIES) {
    auto value = reinterpret_cast<unsigned char *>(param_value);
    value[0] = 1;
  }

  return PI_SUCCESS;
}

static pi_result redefinedProgramBuild(
    pi_program prog, pi_uint32, const pi_device *, const char *,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  if (pfn_notify) {
    pfn_notify(prog, user_data);
  }
  return PI_SUCCESS;
}

static pi_result redefinedKernelCreate(pi_program program,
                                       const char *kernel_name,
                                       pi_kernel *ret_kernel) {
  *ret_kernel = reinterpret_cast<pi_kernel>(new int[1]);
  return PI_SUCCESS;
}

static pi_result redefinedKernelSetExecInfo(pi_kernel kernel,
                                            pi_kernel_exec_info value_name,
                                            size_t param_value_size,
                                            const void *param_value) {
  return PI_SUCCESS;
}

static pi_result redefinedKernelGetInfo(pi_kernel kernel,
                                        pi_kernel_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

static pi_result redefinedKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                                             pi_kernel_group_info param_name,
                                             size_t param_value_size,
                                             void *param_value,
                                             size_t *param_value_size_ret) {
  if (param_name == PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE) {
    if (param_value_size_ret) {
      *param_value_size_ret = 3 * sizeof(size_t);
    } else if (param_value) {
      auto size = static_cast<size_t *>(param_value);
      size[0] = 1;
      size[1] = 1;
      size[2] = 1;
    }
  }

  return PI_SUCCESS;
}

static pi_result redefinedEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                              const size_t *, const size_t *,
                                              const size_t *LocalSize,
                                              pi_uint32 N, const pi_event *Deps,
                                              pi_event *RetEvent) {
  int *Ret = new int[1];
  *Ret = KernelLaunchCounter++;
  printf("Enqueued %i\n", *Ret);

  if (PauseWaitOnIdx == *Ret) {
    // It should be copier kernel. Check if  it depends on user's one.
    EXPECT_EQ(N, 1U);
    int EventIdx = reinterpret_cast<int *>(Deps[0])[0];
    EXPECT_EQ(EventIdx, 0);
  }

  *RetEvent = reinterpret_cast<pi_event>(Ret);
  return PI_SUCCESS;
}

static pi_result redefinedEventsWait(pi_uint32 num_events,
                                     const pi_event *event_list) {
  assert(num_events == 1);

  int EventIdx = reinterpret_cast<int *>(event_list[0])[0];
  printf("Waiting for event %i\n", EventIdx);

  {
    std::lock_guard<std::mutex> Lock{WaitedEventsMutex};
    WaitedEvents.insert(EventIdx);
  }

  if (PauseWaitOnIdx == EventIdx) {
    StartedWait = true;
    while (!ContinueWait)
      ;

    // fail so that host-task isn't going to be executed
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

static pi_result
redefinedMemBufferCreate(pi_context context, pi_mem_flags flags, size_t size,
                         void *host_ptr, pi_mem *ret_mem,
                         const pi_mem_properties *properties = nullptr) {
  *ret_mem = nullptr;
  return PI_SUCCESS;
}

static pi_result redefinedMemRelease(pi_mem mem) { return PI_SUCCESS; }

static pi_result redefinedProgramRetain(pi_program program) {
  return PI_SUCCESS;
}

static pi_result redefinedKernelRetain(pi_kernel kernel) { return PI_SUCCESS; }

static pi_result redefinedKernelRelease(pi_kernel kernel) {
  delete[] reinterpret_cast<int *>(kernel);
  return PI_SUCCESS;
}

static pi_result redefinedKernelSetArg(pi_kernel kernel, pi_uint32 arg_index,
                                       size_t arg_size, const void *arg_value) {
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemBufferMap(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_map,
    pi_map_flags map_flags, size_t offset, size_t size,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event, void **ret_map) {
  return PI_SUCCESS;
}

static void setupMock(sycl::unittest::PiMock &Mock) {
  using namespace sycl::detail;
  Mock.redefine<PiApiKind::piProgramCreate>(redefinedProgramCreate);
  Mock.redefine<PiApiKind::piProgramGetInfo>(redefinedProgramGetInfo);
  Mock.redefine<PiApiKind::piProgramBuild>(redefinedProgramBuild);
  Mock.redefine<PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<PiApiKind::piKernelSetExecInfo>(redefinedKernelSetExecInfo);
  Mock.redefine<PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);
  Mock.redefine<PiApiKind::piKernelGetGroupInfo>(redefinedKernelGetGroupInfo);
  Mock.redefine<PiApiKind::piEnqueueKernelLaunch>(redefinedEnqueueKernelLaunch);
  Mock.redefine<PiApiKind::piMemBufferCreate>(redefinedMemBufferCreate);
  Mock.redefine<PiApiKind::piMemRelease>(redefinedMemRelease);
  Mock.redefine<PiApiKind::piProgramRetain>(redefinedProgramRetain);
  Mock.redefine<PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<PiApiKind::piKernelSetArg>(redefinedKernelSetArg);
  Mock.redefine<PiApiKind::piEnqueueMemBufferMap>(redefinedEnqueueMemBufferMap);
  Mock.redefine<PiApiKind::piEventsWait>(redefinedEventsWait);
}

TEST(Assert, Test) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  sycl::unittest::PiMock Mock{Plt};

  setupMock(Mock);

  std::atomic<bool> ErrorCaptured{false};

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::queue Queue{Dev, [&](sycl::exception_list EL) {
                      for (auto &EPtr : EL)
                        try {
                          std::rethrow_exception(EPtr);
                        } catch (sycl::exception &E) {
                          if (E.get_cl_code() == PI_ERROR_UNKNOWN)
                            ErrorCaptured = true;
                        }
                    }};

  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  auto ExecBundle = sycl::build(KernelBundle);
  Queue.submit([&](sycl::handler &H) {
    H.use_kernel_bundle(ExecBundle);
    H.single_task<TestKernel>([] {});
  });

  while (!StartedWait)
    ;

  ContinueWait = true;

  // Can't return from redefinedEventsWait and report atomically. Hence, here
  // is this wait. Single second wait should be more than enough.
  {
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1000ms);
  }

  Queue.throw_asynchronous();

  while (!ErrorCaptured)
    ;

  // Host-task didn't finish as we returned PI_ERROR_UNKNOWN
  EXPECT_EQ(ErrorCaptured, true);
  // Two kernels to be enqueued: the test kernel and assert info copier
  EXPECT_EQ(KernelLaunchCounter, 2);

  {
    std::lock_guard<std::mutex> Lock{WaitedEventsMutex};
    // Host-task was waiting on the Copier kernel
    EXPECT_EQ(WaitedEvents.count(1) != 0, true);
    EXPECT_EQ(WaitedEvents.size(), 1LU);
  }
}
