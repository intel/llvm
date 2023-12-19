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

#define SYCL_FALLBACK_ASSERT 1
// Enable use of interop kernel c-tor
#define __SYCL_INTERNAL_API
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>

#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

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
static pi_result redefinedKernelGetGroupInfoAfter(
    pi_kernel kernel, pi_device device, pi_kernel_group_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
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

static pi_result
redefinedEnqueueKernelLaunchAfter(pi_queue, pi_kernel, pi_uint32,
                                  const size_t *, const size_t *,
                                  const size_t *LocalSize, pi_uint32 NDeps,
                                  const pi_event *Deps, pi_event *RetEvent) {
  static pi_event UserKernelEvent = *RetEvent;
  int Val = KernelLaunchCounter++;
  // This output here is to reduce amount of time requried to debug/reproduce a
  // failing test upon feature break
  printf("Enqueued %i\n", Val);

  if (PauseWaitOnIdx == Val) {
    // It should be copier kernel. Check if it depends on user's one.
    EXPECT_EQ(NDeps, 1U);
    EXPECT_EQ(Deps[0], UserKernelEvent);
  }

  return PI_SUCCESS;
}

static pi_result redefinedEventsWaitPositive(pi_uint32 num_events,
                                             const pi_event *event_list) {
  // there should be two events: one is for memory map and the other is for
  // copier kernel
  assert(num_events == 2);

  int EventIdx1 = reinterpret_cast<int *>(event_list[0])[0];
  int EventIdx2 = reinterpret_cast<int *>(event_list[1])[0];
  // This output here is to reduce amount of time requried to debug/reproduce
  // a failing test upon feature break
  printf("Waiting for events %i, %i\n", EventIdx1, EventIdx2);
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemBufferMapAfter(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_map,
    pi_map_flags map_flags, size_t offset, size_t size,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *RetEvent, void **RetMap) {
  MemoryMapCounter++;
  // This output here is to reduce amount of time requried to debug/reproduce a
  // failing test upon feature break
  printf("Memory map %i\n", MemoryMapCounter);

  *RetMap = (void *)&ExpectedToOutput;

  return PI_SUCCESS;
}

static void setupMock(sycl::unittest::PiMock &Mock) {
  using namespace sycl::detail;
  Mock.redefineAfter<PiApiKind::piKernelGetGroupInfo>(
      redefinedKernelGetGroupInfoAfter);
  Mock.redefineAfter<PiApiKind::piEnqueueKernelLaunch>(
      redefinedEnqueueKernelLaunchAfter);
  Mock.redefineAfter<PiApiKind::piEnqueueMemBufferMap>(
      redefinedEnqueueMemBufferMapAfter);
  Mock.redefineBefore<PiApiKind::piEventsWait>(redefinedEventsWaitPositive);
}

#ifndef _WIN32
void ChildProcess(int StdErrFD) {
  static constexpr int StandardStdErrFD = 2;
  if (dup2(StdErrFD, StandardStdErrFD) < 0) {
    printf("Can't duplicate stderr fd for %i: %s\n", StdErrFD, strerror(errno));
    exit(1);
  }

  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();

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
  // Ensure that the mock plugin is initialized before spawning work. Since the
  // test needs no redefinitions we do not need to create a PiMock instance, but
  // the mock plugin is still needed to have a valid platform available.
  sycl::unittest::PiMock::EnsureMockPluginInitialized();

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
