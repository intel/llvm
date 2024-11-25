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

#include "ur_mock_helpers.hpp"
#define SYCL_FALLBACK_ASSERT 1
// Enable use of interop kernel c-tor
#define __SYCL_INTERNAL_API
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif // _WIN32

class TestKernel;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<TestKernel> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestKernel"; }
};

static constexpr const kernel_param_desc_t Signatures[] = {
    {kernel_param_kind_t::kind_accessor, 4062, 0}};

template <>
struct KernelInfo<::sycl::detail::__sycl_service_kernel__::AssertInfoCopier>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() {
    return "_ZTSN2cl4sycl6detail23__sycl_service_kernel__16AssertInfoCopierE";
  }
  static constexpr unsigned getNumParams() { return 1; }
  static constexpr const kernel_param_desc_t &getParamDesc(unsigned Idx) {
    assert(!Idx);
    return Signatures[Idx];
  }
  static constexpr int64_t getKernelSize() {
    // The AssertInfoCopier service kernel lambda captures an accessor.
    return sizeof(sycl::accessor<sycl::detail::AssertHappened, 1,
                                 sycl::access::mode::write>);
  }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage generateDefaultImage() {
  using namespace sycl::unittest;

  static const std::string KernelName = "TestKernel";
  static const std::string CopierKernelName =
      "_ZTSN2cl4sycl6detail23__sycl_service_kernel__16AssertInfoCopierE";

  MockPropertySet PropSet;

  setKernelUsesAssert({KernelName}, PropSet);

  std::vector<MockOffloadEntry> Entries = makeEmptyKernels({KernelName});

  MockDeviceImage Img(std::move(Entries), std::move(PropSet));

  return Img;
}

static sycl::unittest::MockDeviceImage generateCopierKernelImage() {
  using namespace sycl::unittest;

  static const std::string CopierKernelName =
      "_ZTSN2cl4sycl6detail23__sycl_service_kernel__16AssertInfoCopierE";

  MockPropertySet PropSet;

  std::vector<MockOffloadEntry> Entries = makeEmptyKernels({CopierKernelName});

  MockDeviceImage Img(std::move(Entries), std::move(PropSet));

  return Img;
}

sycl::unittest::MockDeviceImage Imgs[] = {generateDefaultImage(),
                                          generateCopierKernelImage()};
sycl::unittest::MockDeviceImageArray<2> ImgArray{Imgs};

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
static ur_result_t redefinedKernelGetGroupInfoAfter(void *pParams) {
  auto params = *static_cast<ur_kernel_get_group_info_params_t *>(pParams);
  if (*params.ppropName == UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE) {
    if (*params.ppPropSizeRet) {
      **params.ppPropSizeRet = 3 * sizeof(size_t);
    } else if (*params.ppPropValue) {
      auto size = static_cast<size_t *>(*params.ppPropValue);
      size[0] = 1;
      size[1] = 1;
      size[2] = 1;
    }
  }

  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedEnqueueKernelLaunchAfter(void *pParams) {
  auto params = *static_cast<ur_enqueue_kernel_launch_params_t *>(pParams);
  static ur_event_handle_t UserKernelEvent = **params.pphEvent;
  int Val = KernelLaunchCounter++;
  // This output here is to reduce amount of time requried to debug/reproduce a
  // failing test upon feature break
  printf("Enqueued %i\n", Val);

  if (PauseWaitOnIdx == Val) {
    // It should be copier kernel. Check if it depends on user's one.
    EXPECT_EQ(*params.pnumEventsInWaitList, 1U);
    EXPECT_EQ(*params.pphEventWaitList[0], UserKernelEvent);
  }

  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedEventWaitPositive(void *pParams) {
  auto params = *static_cast<ur_event_wait_params_t *>(pParams);
  // there should be two events: one is for memory map and the other is for
  // copier kernel
  assert(*params.pnumEvents == 2);

  int EventIdx1 = reinterpret_cast<int *>((*params.pphEventWaitList)[0])[0];
  int EventIdx2 = reinterpret_cast<int *>((*params.pphEventWaitList)[1])[0];
  // This output here is to reduce amount of time requried to debug/reproduce
  // a failing test upon feature break
  printf("Waiting for events %i, %i\n", EventIdx1, EventIdx2);
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedEventWaitNegative(void *pParams) {
  auto params = *static_cast<ur_enqueue_events_wait_params_t *>(pParams);
  // For negative tests we do not expect the copier kernel to be used, so
  // instead we accept whatever amount we get.
  // This output here is to reduce amount of time requried to debug/reproduce
  // a failing test upon feature break
  printf("Waiting for %i events ", *params.pnumEventsInWaitList);
  for (size_t I = 0; I < *params.pnumEventsInWaitList; ++I)
    printf("%i, ", reinterpret_cast<int *>(*params.pphEvent[I])[0]);
  printf("\n");
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedEnqueueMemBufferMapAfter(void *pParams) {
  auto params = *static_cast<ur_enqueue_mem_buffer_map_params_t *>(pParams);
  MemoryMapCounter++;
  // This output here is to reduce amount of time requried to debug/reproduce a
  // failing test upon feature break
  printf("Memory map %i\n", MemoryMapCounter);

  **params.pppRetMap = (void *)&ExpectedToOutput;

  return UR_RESULT_SUCCESS;
}

static void setupMock(sycl::unittest::UrMock<> &Mock) {
  using namespace sycl::detail;
  mock::getCallbacks().set_after_callback("urKernelGetGroupInfo",
                                          &redefinedKernelGetGroupInfoAfter);
  mock::getCallbacks().set_after_callback("urEnqueueKernelLaunch",
                                          &redefinedEnqueueKernelLaunchAfter);
  mock::getCallbacks().set_after_callback("urEnqueueMemBufferMap",
                                          &redefinedEnqueueMemBufferMapAfter);
  mock::getCallbacks().set_before_callback("urEventWait",
                                           &redefinedEventWaitPositive);
}

namespace TestInteropKernel {
const sycl::context *Context = nullptr;
const sycl::device *Device = nullptr;
int KernelLaunchCounter = ::KernelLaunchCounterBase;

static ur_result_t redefinedKernelGetInfo(void *pParams) {
  auto params = *static_cast<ur_kernel_get_info_params_t *>(pParams);
  if (UR_KERNEL_INFO_CONTEXT == *params.ppropName) {
    ur_context_handle_t UrContext =
        sycl::detail::getSyclObjImpl(*Context)->getHandleRef();

    if (*params.ppPropValue)
      memcpy(*params.ppPropValue, &UrContext, sizeof(UrContext));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(UrContext);

    return UR_RESULT_SUCCESS;
  }

  if (UR_KERNEL_INFO_PROGRAM == *params.ppropName) {
    ur_program_handle_t URProgram =
        mock::createDummyHandle<ur_program_handle_t>();

    if (*params.ppPropValue)
      memcpy(*params.ppPropValue, &URProgram, sizeof(URProgram));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(URProgram);

    return UR_RESULT_SUCCESS;
  }

  if (UR_KERNEL_INFO_FUNCTION_NAME == *params.ppropName) {
    static const char FName[] = "TestFnName";
    if (*params.ppPropValue) {
      size_t L = strlen(FName) + 1;
      if (L < *params.ppropSize)
        L = *params.ppropSize;

      memcpy(*params.ppPropValue, FName, L);
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = strlen(FName) + 1;

    return UR_RESULT_SUCCESS;
  }

  return UR_RESULT_ERROR_UNKNOWN;
}

static ur_result_t redefinedEnqueueKernelLaunch(void *pParms) {
  int Val = KernelLaunchCounter++;
  // This output here is to reduce amount of time requried to debug/reproduce a
  // failing test upon feature break
  printf("Enqueued %i\n", Val);

  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedProgramGetInfo(void *pParams) {
  auto params = *static_cast<ur_program_get_info_params_t *>(pParams);
  if (UR_PROGRAM_INFO_NUM_DEVICES == *params.ppropName) {
    static const int V = 1;

    if (*params.ppPropValue)
      memcpy(*params.ppPropValue, &V, sizeof(V));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(V);

    return UR_RESULT_SUCCESS;
  }

  if (UR_PROGRAM_INFO_DEVICES == *params.ppropName) {
    EXPECT_EQ(*params.ppropSize, 1 * sizeof(ur_device_handle_t));

    ur_device_handle_t Dev = sycl::detail::getSyclObjImpl(*Device)->getHandleRef();

    if (*params.ppPropValue)
      memcpy(*params.ppPropValue, &Dev, sizeof(Dev));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(Dev);

    return UR_RESULT_SUCCESS;
  }

  // Required if program cache eviction is enabled.
  if (UR_PROGRAM_INFO_BINARY_SIZES == *params.ppropName) {
    size_t BinarySize = 1;

    if (*params.ppPropValue)
      memcpy(*params.ppPropValue, &BinarySize, sizeof(size_t));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(size_t);

    return UR_RESULT_SUCCESS;
  }

  return UR_RESULT_ERROR_UNKNOWN;
}

static ur_result_t redefinedProgramGetBuildInfo(void *pParams) {
  auto params = *static_cast<ur_program_get_build_info_params_t *>(pParams);
  if (UR_PROGRAM_BUILD_INFO_BINARY_TYPE == *params.ppropName) {
    static const ur_program_binary_type_t T = UR_PROGRAM_BINARY_TYPE_EXECUTABLE;
    if (*params.ppPropValue)
      memcpy(*params.ppPropValue, &T, sizeof(T));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(T);
    return UR_RESULT_SUCCESS;
  }

  if (UR_PROGRAM_BUILD_INFO_OPTIONS == *params.ppropName) {
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = 0;
    return UR_RESULT_SUCCESS;
  }

  return UR_RESULT_ERROR_UNKNOWN;
}

} // namespace TestInteropKernel

static void setupMockForInterop(sycl::unittest::UrMock<> &Mock,
                                const sycl::context &Ctx,
                                const sycl::device &Dev) {
  using namespace sycl::detail;

  TestInteropKernel::KernelLaunchCounter = ::KernelLaunchCounterBase;
  TestInteropKernel::Device = &Dev;
  TestInteropKernel::Context = &Ctx;

  mock::getCallbacks().set_after_callback("urKernelGetGroupInfo",
                                          &redefinedKernelGetGroupInfoAfter);
  mock::getCallbacks().set_before_callback(
      "urEnqueueKernelLaunch",
      &TestInteropKernel::redefinedEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urEnqueueMemBufferMap",
                                          &redefinedEnqueueMemBufferMapAfter);
  mock::getCallbacks().set_before_callback("urEventWait",
                                           &redefinedEventWaitNegative);
  mock::getCallbacks().set_before_callback(
      "urKernelGetInfo", &TestInteropKernel::redefinedKernelGetInfo);
  mock::getCallbacks().set_before_callback(
      "urProgramGetInfo", &TestInteropKernel::redefinedProgramGetInfo);
  mock::getCallbacks().set_before_callback(
      "urProgramGetBuildInfo",
      &TestInteropKernel::redefinedProgramGetBuildInfo);
}

#ifndef _WIN32
void ChildProcess(int StdErrFD) {
  static constexpr int StandardStdErrFD = 2;
  if (dup2(StdErrFD, StandardStdErrFD) < 0) {
    printf("Can't duplicate stderr fd for %i: %s\n", StdErrFD, strerror(errno));
    exit(1);
  }

  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  setupMock(Mock);

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

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

TEST(Assert, TestPositive) {
  // Ensure that the mock adapter is initialized before spawning work. Since the
  // test needs no redefinitions we do not need to create a UrMock<> instance,
  // but the mock adapter is still needed to have a valid platform available.
  // sycl::unittest::UrMock::InitUr();

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

TEST(Assert, TestAssertServiceKernelHidden) {
  const char *AssertServiceKernelName = sycl::detail::KernelInfo<
      sycl::detail::__sycl_service_kernel__::AssertInfoCopier>::getName();

  std::vector<sycl::kernel_id> AllKernelIDs = sycl::get_kernel_ids();

  auto NoFoundServiceKernelID = std::none_of(
      AllKernelIDs.begin(), AllKernelIDs.end(), [=](sycl::kernel_id KernelID) {
        return strcmp(KernelID.get_name(), AssertServiceKernelName) == 0;
      });

  EXPECT_TRUE(NoFoundServiceKernelID);
}

TEST(Assert, TestInteropKernelNegative) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};

  setupMockForInterop(Mock, Ctx, Dev);

  sycl::queue Queue{Ctx, Dev};

  auto URKernel = mock::createDummyHandle<ur_kernel_handle_t>();

  // TODO use make_kernel. This requires a fix in backend.cpp to get adapter
  // from context instead of free getAdapter to allow for mocking of its
  // methods
  sycl::kernel KInterop((cl_kernel)URKernel, Ctx);

  Queue.submit([&](sycl::handler &H) { H.single_task(KInterop); });

  EXPECT_EQ(TestInteropKernel::KernelLaunchCounter,
            KernelLaunchCounterBase + 1);
}

TEST(Assert, TestInteropKernelFromProgramNegative) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};

  setupMockForInterop(Mock, Ctx, Dev);

  sycl::queue Queue{Ctx, Dev};

  sycl::kernel_bundle Bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx);
  sycl::kernel KOrig = Bundle.get_kernel(sycl::get_kernel_id<TestKernel>());

  cl_kernel CLKernel = sycl::get_native<sycl::backend::opencl>(KOrig);
  sycl::kernel KInterop{CLKernel, Ctx};

  Queue.submit([&](sycl::handler &H) { H.single_task(KInterop); });

  EXPECT_EQ(TestInteropKernel::KernelLaunchCounter,
            KernelLaunchCounterBase + 1);
}
