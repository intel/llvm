//==---------- assert.cpp --- Check assert helpers enqueue -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
 * The positive test here checks that assert fallback assert feature works well.
 * According to the doc, when assert is triggered on device host application
 * should abort. That said, a standard `abort()` function is to be called. The
 * function makes sure the app terminates due `SIGABRT` signal. This makes it
 * impossible to verify the feature in uni-process environment. Hence, we employ
 * multi-process envirnment i.e. we call a `fork()`. The child process is should
 * abort and the parent process verifies it and checks that child prints correct
 * error message to `stderr`. Verification of `stderr` output is performed via
 * pipe.
 */

// Enable use of interop kernel c-tor
#define __SYCL_INTERNAL_API
#include <CL/sycl.hpp>
#include <CL/sycl/backend/opencl.hpp>

#include <helpers/PiImage.hpp>
#include <helpers/sycl_test.hpp>

#include <gtest/gtest.h>

#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif // _WIN32

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

template <>
struct KernelInfo<::sycl::detail::__sycl_service_kernel__::AssertInfoCopier> {
  static constexpr const char *getName() {
    return "_ZTSN2cl4sycl6detail23__sycl_service_kernel__16AssertInfoCopierE";
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
      "_ZTSN2cl4sycl6detail23__sycl_service_kernel__16AssertInfoCopierE";

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
      "_ZTSN2cl4sycl6detail23__sycl_service_kernel__16AssertInfoCopierE";

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

struct AssertHappened {
  int Flag = 0;
  char Expr[256 + 1] = "";
  char File[256 + 1] = "";
  char Func[128 + 1] = "";

  int32_t Line = 0;

  uint64_t GID0 = 0;
  uint64_t GID1 = 0;
  uint64_t GID2 = 0;

  uint64_t LID0 = 0;
  uint64_t LID1 = 0;
  uint64_t LID2 = 0;
};

// This should not be modified.
// Substituted in memory map operation.
static AssertHappened ExpectedToOutput = {
    2, // assert copying done
    "TestExpression",
    "TestFile",
    "TestFunc",
    123, // line

    0, // global id
    1, // global id
    2, // global id
    3, // local id
    4, // local id
    5  // local id
};

static constexpr int KernelLaunchCounterBase = 0;
static int KernelLaunchCounter = KernelLaunchCounterBase;
static constexpr int MemoryMapCounterBase = 1000;
static int MemoryMapCounter = MemoryMapCounterBase;
static constexpr int PauseWaitOnIdx = KernelLaunchCounterBase + 1;

// Mock redifinitions
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

static pi_result redefinedEnqueueMemBufferRead(
    pi_queue queue, pi_mem buffer, pi_bool blocking_read, size_t offset,
    size_t size, void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  *event = reinterpret_cast<pi_event>(new size_t{1});
  return PI_SUCCESS;
}

static pi_result redefinedProgramCreateWithSource(pi_context context,
                                                  pi_uint32 count,
                                                  const char **strings,
                                                  const size_t *lengths,
                                                  pi_program *ret_program) {
  *ret_program = reinterpret_cast<pi_program>(new size_t{1});
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                              const size_t *, const size_t *,
                                              const size_t *LocalSize,
                                              pi_uint32 N, const pi_event *Deps,
                                              pi_event *RetEvent) {
  int *Ret = new int[1];
  *Ret = KernelLaunchCounter++;
  // This output here is to reduce amount of time requried to debug/reproduce a
  // failing test upon feature break
  printf("Enqueued %i\n", *Ret);

  if (PauseWaitOnIdx == *Ret) {
    // It should be copier kernel. Check if it depends on user's one.
    EXPECT_EQ(N, 1U);
    int EventIdx = reinterpret_cast<int *>(Deps[0])[0];
    EXPECT_EQ(EventIdx, 0);
  }

  *RetEvent = reinterpret_cast<pi_event>(Ret);
  return PI_SUCCESS;
}

static pi_result redefinedEventsWait(pi_uint32 num_events,
                                     const pi_event *event_list) {
  // there should be two events: one is for memory map and the other is for
  // copier kernel
  assert(num_events == 2);

  int EventIdx1 = reinterpret_cast<int *>(event_list[0])[0];
  int EventIdx2 = reinterpret_cast<int *>(event_list[1])[0];
  // This output here is to reduce amount of time requried to debug/reproduce a
  // failing test upon feature break
  printf("Waiting for events %i, %i\n", EventIdx1, EventIdx2);
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

static pi_result redefinedKernelSetArg(pi_kernel kernel, pi_uint32 arg_index,
                                       size_t arg_size, const void *arg_value) {
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemBufferMap(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_map,
    pi_map_flags map_flags, size_t offset, size_t size,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *RetEvent, void **RetMap) {
  int *Ret = new int[1];
  *Ret = MemoryMapCounter++;
  // This output here is to reduce amount of time requried to debug/reproduce a
  // failing test upon feature break
  printf("Memory map %i\n", *Ret);
  *RetEvent = reinterpret_cast<pi_event>(Ret);

  *RetMap = (void *)&ExpectedToOutput;

  return PI_SUCCESS;
}

static pi_result redefinedExtKernelSetArgMemObj(pi_kernel kernel,
                                                pi_uint32 arg_index,
                                                const pi_mem *arg_value) {
  return PI_SUCCESS;
}

static void setupMock() {
  using namespace sycl::detail;
  using namespace sycl::unittest;

  redefine<PiApiKind::piKernelGetGroupInfo>(redefinedKernelGetGroupInfo);
  redefine<PiApiKind::piEnqueueKernelLaunch>(redefinedEnqueueKernelLaunch);
  redefine<PiApiKind::piMemBufferCreate>(redefinedMemBufferCreate);
  redefine<PiApiKind::piMemRelease>(redefinedMemRelease);
  redefine<PiApiKind::piKernelSetArg>(redefinedKernelSetArg);
  redefine<PiApiKind::piEnqueueMemBufferMap>(redefinedEnqueueMemBufferMap);
  redefine<PiApiKind::piEventsWait>(redefinedEventsWait);
  redefine<PiApiKind::piextKernelSetArgMemObj>(redefinedExtKernelSetArgMemObj);
  redefine<PiApiKind::piclProgramCreateWithSource>(
      redefinedProgramCreateWithSource);
}

namespace TestInteropKernel {
const sycl::context *Context = nullptr;
const sycl::device *Device = nullptr;
int KernelLaunchCounter = ::KernelLaunchCounterBase;

static pi_result redefinedKernelGetInfo(pi_kernel Kernel,
                                        pi_kernel_info ParamName,
                                        size_t ParamValueSize, void *ParamValue,
                                        size_t *ParamValueSizeRet) {
  if (PI_KERNEL_INFO_CONTEXT == ParamName) {
    cl_context Ctx = sycl::get_native<sycl::backend::opencl>(*Context);

    if (ParamValue)
      memcpy(ParamValue, &Ctx, sizeof(Ctx));
    if (ParamValueSizeRet)
      *ParamValueSizeRet = sizeof(Ctx);

    return PI_SUCCESS;
  }

  if (PI_KERNEL_INFO_PROGRAM == ParamName) {
    cl_program X = (cl_program)1;

    if (ParamValue)
      memcpy(ParamValue, &X, sizeof(X));
    if (ParamValueSizeRet)
      *ParamValueSizeRet = sizeof(X);

    return PI_SUCCESS;
  }

  if (sycl::info::kernel::function_name == (sycl::info::kernel)ParamName) {
    static const char FName[] = "TestFnName";
    if (ParamValue) {
      size_t L = strlen(FName) + 1;
      if (L < ParamValueSize)
        L = ParamValueSize;

      memcpy(ParamValue, FName, L);
    }
    if (ParamValueSizeRet)
      *ParamValueSizeRet = strlen(FName) + 1;

    return PI_SUCCESS;
  }

  return PI_ERROR_UNKNOWN;
}

static pi_result redefinedEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                              const size_t *, const size_t *,
                                              const size_t *LocalSize,
                                              pi_uint32 N, const pi_event *Deps,
                                              pi_event *RetEvent) {
  int *Ret = new int[1];
  *Ret = KernelLaunchCounter++;
  // This output here is to reduce amount of time requried to debug/reproduce a
  // failing test upon feature break
  printf("Enqueued %i\n", *Ret);

  *RetEvent = reinterpret_cast<pi_event>(Ret);
  return PI_SUCCESS;
}

static pi_result redefinedProgramGetInfo(pi_program P,
                                         pi_program_info ParamName,
                                         size_t ParamValueSize,
                                         void *ParamValue,
                                         size_t *ParamValueSizeRet) {
  if (PI_PROGRAM_INFO_NUM_DEVICES == ParamName) {
    static const int V = 1;

    if (ParamValue)
      memcpy(ParamValue, &V, sizeof(V));
    if (ParamValueSizeRet)
      *ParamValueSizeRet = sizeof(V);

    return PI_SUCCESS;
  }

  if (PI_PROGRAM_INFO_DEVICES == ParamName) {
    EXPECT_EQ(ParamValueSize, 1 * sizeof(cl_device_id));

    cl_device_id Dev = sycl::get_native<sycl::backend::opencl>(*Device);

    if (ParamValue)
      memcpy(ParamValue, &Dev, sizeof(Dev));
    if (ParamValueSizeRet)
      *ParamValueSizeRet = sizeof(Dev);

    return PI_SUCCESS;
  }

  return PI_ERROR_UNKNOWN;
}

static pi_result redefinedProgramGetBuildInfo(
    pi_program P, pi_device D,
    cl_program_build_info ParamName, // TODO: untie from OpenCL
    size_t ParamValueSize, void *ParamValue, size_t *ParamValueSizeRet) {
  if (CL_PROGRAM_BINARY_TYPE == ParamName) {
    static const cl_program_binary_type T = CL_PROGRAM_BINARY_TYPE_EXECUTABLE;
    if (ParamValue)
      memcpy(ParamValue, &T, sizeof(T));
    if (ParamValueSizeRet)
      *ParamValueSizeRet = sizeof(T);
    return PI_SUCCESS;
  }

  if (CL_PROGRAM_BUILD_OPTIONS == ParamName) {
    if (ParamValueSizeRet)
      *ParamValueSizeRet = 0;
    return PI_SUCCESS;
  }

  return PI_ERROR_UNKNOWN;
}

} // namespace TestInteropKernel

static void setupMockForInterop(const sycl::context &Ctx,
                                const sycl::device &Dev) {
  using namespace sycl::detail;
  using namespace sycl::unittest;

  TestInteropKernel::KernelLaunchCounter = ::KernelLaunchCounterBase;
  TestInteropKernel::Device = &Dev;
  TestInteropKernel::Context = &Ctx;

  redefine<PiApiKind::piKernelGetGroupInfo>(redefinedKernelGetGroupInfo);
  redefine<PiApiKind::piEnqueueKernelLaunch>(
      TestInteropKernel::redefinedEnqueueKernelLaunch);
  redefine<PiApiKind::piMemBufferCreate>(redefinedMemBufferCreate);
  redefine<PiApiKind::piMemRelease>(redefinedMemRelease);
  redefine<PiApiKind::piKernelSetArg>(redefinedKernelSetArg);
  redefine<PiApiKind::piEnqueueMemBufferMap>(redefinedEnqueueMemBufferMap);
  redefine<PiApiKind::piEventsWait>(redefinedEventsWait);
  redefine<PiApiKind::piextKernelSetArgMemObj>(redefinedExtKernelSetArgMemObj);
  redefine<PiApiKind::piKernelGetInfo>(
      TestInteropKernel::redefinedKernelGetInfo);
  redefine<PiApiKind::piProgramGetInfo>(
      TestInteropKernel::redefinedProgramGetInfo);
  redefine<PiApiKind::piProgramGetBuildInfo>(
      TestInteropKernel::redefinedProgramGetBuildInfo);
  redefine<PiApiKind::piclProgramCreateWithSource>(
      redefinedProgramCreateWithSource);
  redefine<PiApiKind::piEnqueueMemBufferRead>(redefinedEnqueueMemBufferRead);
}

#ifndef _WIN32
void ChildProcess(int StdErrFD) {
  static constexpr int StandardStdErrFD = 2;
  if (dup2(StdErrFD, StandardStdErrFD) < 0) {
    printf("Can't duplicate stderr fd for %i: %s\n", StdErrFD, strerror(errno));
    exit(1);
  }

  sycl::platform Plt{sycl::default_selector()};

  setupMock();

  const sycl::device Dev = Plt.get_devices()[0];
  const sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  auto ExecBundle = sycl::build(KernelBundle);
  printf("Child process launching kernel\n");
  Queue.submit([&](sycl::handler &H) {
    H.use_kernel_bundle(ExecBundle);
    H.single_task<TestKernel>([] {});
  });
  printf("Child process waiting on the queue\n");
  Queue.wait();
  printf("Child process done waiting on the queue. That's unexpected\n");
  exit(1);
}

void ParentProcess(int ChildPID, int ChildStdErrFD) {
  static constexpr char StandardMessage[] =
      "TestFile:123: TestFunc: global id:"
      " [0,1,2], local id: [3,4,5] Assertion `TestExpression` failed.";

  int Status = 0;

  printf("Parent process waiting for child %i\n", ChildPID);

  waitpid(ChildPID, &Status, /*options = */ 0);

  int SigNum = WTERMSIG(Status);

  // Fetch number of unread bytes in pipe
  int PipeUnread = 0;
  if (ioctl(ChildStdErrFD, FIONREAD, &PipeUnread) < 0) {
    perror("Couldn't fetch pipe size: ");
    exit(1);
  }

  std::vector<char> Buf(PipeUnread + 1, '\0');

  // Read the pipe contents
  {
    size_t TotalReadCnt = 0;

    while (TotalReadCnt < static_cast<size_t>(PipeUnread)) {
      ssize_t ReadCnt = read(ChildStdErrFD, Buf.data() + TotalReadCnt,
                             PipeUnread - TotalReadCnt);

      if (ReadCnt < 0) {
        perror("Couldn't read from pipe");
        exit(1);
      }

      TotalReadCnt += ReadCnt;
    }
  }

  std::string BufStr(Buf.data());

  printf("Status: %i, Signal: %i, Buffer: >>> %s <<<\n", Status, SigNum,
         Buf.data());

  EXPECT_EQ(!!WIFSIGNALED(Status), true);
  EXPECT_EQ(SigNum, SIGABRT);
  EXPECT_NE(BufStr.find(StandardMessage), std::string::npos);
}
#endif // _WIN32

SYCL_TEST(Assert, TestPositive) {
  // Preliminary checks
  {
    sycl::platform Plt{sycl::default_selector()};
    if (Plt.is_host()) {
      printf("Test is not supported on host, skipping\n");
      return;
    }

    if (Plt.get_backend() == sycl::backend::ext_oneapi_cuda) {
      printf("Test is not supported on CUDA platform, skipping\n");
      return;
    }

    if (Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
      printf("Test is not supported on HIP platform, skipping\n");
      return;
    }
  }

#ifndef _WIN32
  static constexpr int ReadFDIdx = 0;
  static constexpr int WriteFDIdx = 1;
  int PipeFD[2];

  if (pipe(PipeFD) < 0) {
    perror("Failed to create pipe for stderr: ");
    exit(1);
  }

  int ChildPID = fork();

  if (ChildPID) {
    close(PipeFD[WriteFDIdx]);
    ParentProcess(ChildPID, PipeFD[ReadFDIdx]);
    close(PipeFD[ReadFDIdx]);
  } else {
    close(PipeFD[ReadFDIdx]);
    ChildProcess(PipeFD[WriteFDIdx]);
    close(PipeFD[WriteFDIdx]);
  }
#endif // _WIN32
}

SYCL_TEST(Assert, TestAssertServiceKernelHidden) {
  const char *AssertServiceKernelName = sycl::detail::KernelInfo<
      sycl::detail::__sycl_service_kernel__::AssertInfoCopier>::getName();

  std::vector<sycl::kernel_id> AllKernelIDs = sycl::get_kernel_ids();

  auto NoFoundServiceKernelID = std::none_of(
      AllKernelIDs.begin(), AllKernelIDs.end(), [=](sycl::kernel_id KernelID) {
        return strcmp(KernelID.get_name(), AssertServiceKernelName) == 0;
      });

  EXPECT_TRUE(NoFoundServiceKernelID);
}

SYCL_TEST(Assert, TestInteropKernelNegative) {
  sycl::platform Plt{sycl::default_selector()};

  if (Plt.is_host()) {
    printf("Test is not supported on host, skipping\n");
    return;
  }

  const sycl::backend Backend = Plt.get_backend();

  if (Backend == sycl::backend::ext_oneapi_cuda ||
      Backend == sycl::backend::ext_oneapi_hip ||
      Backend == sycl::backend::ext_oneapi_level_zero) {
    printf(
        "Test is not supported on CUDA, HIP, Level Zero platforms, skipping\n");
    return;
  }

  const sycl::device Dev = Plt.get_devices()[0];
  const sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  setupMockForInterop(Ctx, Dev);

  cl_kernel CLKernel = (cl_kernel)(0x01);
  // TODO use make_kernel. This requires a fix in backend.cpp to get plugin
  // from context instead of free getPlugin to alllow for mocking of its methods
  sycl::kernel KInterop(CLKernel, Ctx);

  Queue.submit([&](sycl::handler &H) { H.single_task(KInterop); });

  EXPECT_EQ(TestInteropKernel::KernelLaunchCounter,
            KernelLaunchCounterBase + 1);
}

SYCL_TEST(Assert, TestInteropKernelFromProgramNegative) {
  sycl::platform Plt{sycl::default_selector()};

  if (Plt.is_host()) {
    printf("Test is not supported on host, skipping\n");
    return;
  }

  const sycl::backend Backend = Plt.get_backend();

  if (Backend == sycl::backend::ext_oneapi_cuda ||
      Backend == sycl::backend::ext_oneapi_hip ||
      Backend == sycl::backend::ext_oneapi_level_zero) {
    printf(
        "Test is not supported on CUDA, HIP, Level Zero platforms, skipping\n");
    return;
  }

  const sycl::device Dev = Plt.get_devices()[0];
  const sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  setupMockForInterop(Ctx, Dev);

  sycl::kernel_bundle Bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx);
  sycl::kernel KOrig = Bundle.get_kernel(sycl::get_kernel_id<TestKernel>());

  cl_kernel CLKernel = sycl::get_native<sycl::backend::opencl>(KOrig);
  sycl::kernel KInterop{CLKernel, Ctx};

  Queue.submit([&](sycl::handler &H) { H.single_task(KInterop); });

  EXPECT_EQ(TestInteropKernel::KernelLaunchCounter,
            KernelLaunchCounterBase + 1);
}

SYCL_TEST(Assert, DISABLED_TestKernelFromSourceNegative) {
  sycl::platform Plt{sycl::default_selector()};

  if (Plt.is_host()) {
    printf("Test is not supported on host, skipping\n");
    return;
  }

  const sycl::backend Backend = Plt.get_backend();

  if (Backend == sycl::backend::ext_oneapi_cuda ||
      Backend == sycl::backend::ext_oneapi_hip ||
      Backend == sycl::backend::ext_oneapi_level_zero) {
    printf(
        "Test is not supported on CUDA, HIP, Level Zero platforms, skipping\n");
    return;
  }

  constexpr size_t Size = 16;
  std::array<int, Size> Data;

  for (size_t I = 0; I < Size; I++) {
    Data[I] = I;
  }

  sycl::buffer<int, 1> Buf{Data};

  const sycl::device Dev = Plt.get_devices()[0];
  const sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  setupMockForInterop(Ctx, Dev);

  sycl::program P{Ctx};
  P.build_with_source(R"CLC(
          kernel void add(global int* data) {
              int index = get_global_id(0);
              data[index] = data[index] + 1;
          }
      )CLC",
                      "-cl-fast-relaxed-math");

  Queue.submit([&](sycl::handler &H) {
    auto Acc = Buf.get_access<sycl::access::mode::read_write>(H);

    H.set_args(Acc);
    H.parallel_for(Size, P.get_kernel("add"));
  });

  EXPECT_EQ(TestInteropKernel::KernelLaunchCounter,
            KernelLaunchCounterBase + 1);
}
