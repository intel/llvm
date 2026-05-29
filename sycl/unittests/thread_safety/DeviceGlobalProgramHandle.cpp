//==- DeviceGlobalProgramHandle.cpp -- Thread safety of program handles ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests that program handles used in enqueueDeviceGlobal operations are read
// safely without holding a Program lock.  The correctness of the lock-free
// access relies on two invariants:
//
//   1. KernelProgramCache::getOrBuild() writes BuildResult::Val before calling
//      updateAndNotify(BS_Done), which acquires MBuildResultMutex before
//      storing the new state.  waitUntilTransition() holds the same mutex
//      while waiting, establishing a happens-before edge: the write of Val
//      happens-before any read of Val that follows a successful
//      waitUntilTransition() call.
//
//   2. BuildResult::Val is immutable once the build state is BS_Done; no code
//      path overwrites it afterwards.  Therefore concurrent lock-free reads
//      by multiple threads after BS_Done are safe.
//
// The test exercises the code path in memcpyToDeviceGlobalDirect (called via
// queue::copy on a device_image_scope device_global), where
// getOrBuildProgramForDeviceGlobal() is invoked and the returned
// ur_program_handle_t is forwarded to urEnqueueDeviceGlobalVariableWrite
// without any additional locking.  memcpyFromDeviceGlobalDirect also calls
// the same getOrBuildProgramForDeviceGlobal helper, but this test validates
// only the write direction.
//
//===----------------------------------------------------------------------===//

#include "ThreadUtils.h"
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

//===----------------------------------------------------------------------===//
// Kernel class and device_global declaration
//
// The device_global carries device_image_scope so that copy operations go
// through memcpyToDeviceGlobalDirect / memcpyFromDeviceGlobalDirect.  That
// path retrieves the ur_program_handle_t from the cache (via
// getOrBuildProgramForDeviceGlobal) and passes it directly to the UR enqueue
// calls – the exact scenario under scrutiny.
//===----------------------------------------------------------------------===//

class DevGlobHandleTestKernel;
constexpr const char *DevGlobHandleTestKernelName = "DevGlobHandleTestKernel";
constexpr const char *DevGlobHandleTestGlobalName = "DevGlobHandleTestGlobal";

using DevGlobElem = int[2];

sycl::ext::oneapi::experimental::device_global<
    DevGlobElem, decltype(sycl::ext::oneapi::experimental::properties(
                     sycl::ext::oneapi::experimental::device_image_scope))>
    g_TestDevGlobal;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<DevGlobHandleTestKernel>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return DevGlobHandleTestKernelName; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage generateHandleTestImage() {
  using namespace sycl::unittest;

  sycl::detail::device_global_map::add(&g_TestDevGlobal,
                                       DevGlobHandleTestGlobalName);

  MockPropertySet PropSet;
  MockProperty DevGlobInfo =
      makeDeviceGlobalInfo(DevGlobHandleTestGlobalName, sizeof(DevGlobElem),
                           /*IsDeviceImageScope=*/1);
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_DEVICE_GLOBALS,
                 std::vector<MockProperty>{std::move(DevGlobInfo)});

  std::vector<MockOffloadEntry> Entries =
      makeEmptyKernels({DevGlobHandleTestKernelName});

  return MockDeviceImage(std::move(Entries), std::move(PropSet));
}

namespace {

sycl::unittest::MockDeviceImage g_Imgs[] = {generateHandleTestImage()};
sycl::unittest::MockDeviceImageArray<1> g_ImgArray{g_Imgs};

//===----------------------------------------------------------------------===//
// Shared mock state
//
// All state is reset in the fixture's SetUp() before each test.
//===----------------------------------------------------------------------===//

std::mutex g_HandlesMtx;
std::vector<ur_program_handle_t> g_CapturedHandles;

// Gate used by CopyToDeviceGlobalBuildRace to control urProgramBuildExp.
//
// g_BuildBlockMtx protects both g_BuildCanProceed and g_BuildStarted so that
// both directions of the handshake use the same lock, avoiding the need for
// a separate synchronisation primitive for each flag.
std::mutex g_BuildBlockMtx;
std::condition_variable g_BuildBlockCV;
bool g_BuildCanProceed = false;
bool g_BuildStarted = false;

//===----------------------------------------------------------------------===//
// Mock callbacks
//===----------------------------------------------------------------------===//

/// Captures the program handle received by urEnqueueDeviceGlobalVariableWrite.
static ur_result_t after_urEnqueueDeviceGlobalVariableWrite(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_device_global_variable_write_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(g_HandlesMtx);
  g_CapturedHandles.push_back(*params.phProgram);
  return UR_RESULT_SUCCESS;
}

/// Replaces urProgramBuildExp to block until g_BuildCanProceed is set.
/// This simulates a slow program build so that concurrent copy calls encounter
/// the program in BS_InProgress state and are forced through
/// waitUntilTransition().
static ur_result_t blocking_urProgramBuildExp(void *) {
  {
    std::lock_guard<std::mutex> Lock(g_BuildBlockMtx);
    g_BuildStarted = true;
  }
  // Notify the main thread that the build has started, then wait for it to
  // release the gate. Both signals share g_BuildBlockCV so the main thread
  // only needs to wait on one condition variable.
  g_BuildBlockCV.notify_all();
  std::unique_lock<std::mutex> Lock(g_BuildBlockMtx);
  g_BuildBlockCV.wait(Lock, [] { return g_BuildCanProceed; });
  return UR_RESULT_SUCCESS;
}

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class DeviceGlobalProgramHandleTest : public ::testing::Test {
protected:
  void SetUp() override {
    {
      std::lock_guard<std::mutex> HLock(g_HandlesMtx);
      g_CapturedHandles.clear();
    }
    {
      std::lock_guard<std::mutex> BLock(g_BuildBlockMtx);
      g_BuildStarted = false;
      g_BuildCanProceed = false;
    }

    sycl::platform Plt = sycl::platform();
    Q = sycl::queue(Plt.get_devices()[0]);
  }

public:
  sycl::unittest::UrMock<> Mock;
  sycl::queue Q;
};

//===----------------------------------------------------------------------===//
// Test 1 – CopyToDeviceGlobalBuildRace
//
// One thread blocks inside urProgramBuildExp (simulating a slow backend
// build) while the remaining NThreads-1 threads enter getOrBuild(), find
// the state BS_InProgress, and block inside waitUntilTransition().
//
// Once the build gate is released:
//   - BuildResult::Val is written (the ur_program_handle_t).
//   - updateAndNotify(BS_Done) acquires MBuildResultMutex, stores BS_Done,
//     releases the mutex, and calls notify_all().
//   - Every waiting thread wakes up, re-acquires MBuildResultMutex to
//     confirm the predicate, then returns – establishing the happens-before
//     edge between the Val write and the subsequent lock-free Val read in
//     memcpyToDeviceGlobalDirect.
//
// The test asserts that every thread passes a valid, identical handle to
// urEnqueueDeviceGlobalVariableWrite, confirming that reading the handle
// without a lock is correct for the write direction.
//===----------------------------------------------------------------------===//

TEST_F(DeviceGlobalProgramHandleTest, CopyToDeviceGlobalBuildRace) {
  mock::getCallbacks().set_after_callback(
      "urEnqueueDeviceGlobalVariableWrite",
      &after_urEnqueueDeviceGlobalVariableWrite);
  mock::getCallbacks().set_replace_callback("urProgramBuildExp",
                                            &blocking_urProgramBuildExp);

  constexpr std::size_t NThreads = 4;
  Barrier StartBarrier(NThreads);
  int Vals[2] = {42, 1234};

  // All threads start simultaneously. Exactly one wins the
  // compare_exchange_strong in getOrBuild() and enters urProgramBuildExp
  // (where it blocks). The remaining NThreads-1 threads find the state
  // BS_InProgress and block inside waitUntilTransition().
  auto Task = [&](std::size_t /*ThreadId*/) {
    StartBarrier.wait();
    Q.copy(Vals, g_TestDevGlobal).wait();
  };

  // Launch the pool on a separate thread so the main thread stays free to
  // control the build gate.
  std::thread PoolRunner([&]() { ThreadPool Pool(NThreads, Task); });

  // Scope guard: unconditionally release the build gate and join PoolRunner
  // when this scope exits, regardless of whether the test body reaches the
  // normal release/join point below.  Without this, a failed ASSERT_* would:
  //   - leave worker threads blocked on g_BuildBlockCV forever, and
  //   - leave PoolRunner joinable, causing std::terminate() in its destructor.
  // On the normal path the explicit join below runs first; the guard's join
  // is then a no-op because joinable() returns false.
  struct GateAndThreadGuard {
    std::thread &Runner;
    ~GateAndThreadGuard() {
      {
        std::lock_guard<std::mutex> Lock(g_BuildBlockMtx);
        g_BuildCanProceed = true;
      }
      g_BuildBlockCV.notify_all();
      if (Runner.joinable())
        Runner.join();
    }
  } Guard{PoolRunner};

  // Wait for the builder thread to signal it has entered urProgramBuildExp.
  // A timed wait is used so the test fails with a clear message instead of
  // hanging indefinitely if the callback is never reached (e.g. because the
  // program was already cached or the replace callback was not installed).
  constexpr auto BuildStartTimeout = std::chrono::seconds(5);
  bool BuildStarted = false;
  {
    std::unique_lock<std::mutex> Lock(g_BuildBlockMtx);
    BuildStarted = g_BuildBlockCV.wait_for(Lock, BuildStartTimeout,
                                           [] { return g_BuildStarted; });
  }
  ASSERT_TRUE(BuildStarted)
      << "Timed out waiting for urProgramBuildExp to start; the blocking "
         "mock was not called, which means the program may have been served "
         "from cache or the replace callback was not installed correctly";

  // Release the build. notify_all() is called outside the lock to avoid
  // waking the builder thread only for it to immediately block again on
  // re-acquiring the mutex.
  {
    std::lock_guard<std::mutex> Lock(g_BuildBlockMtx);
    g_BuildCanProceed = true;
  }
  g_BuildBlockCV.notify_all();

  // Wait for all worker threads to finish before inspecting the captured
  // handles.  PoolRunner is also joined by the GateGuard on early exit to
  // prevent std::terminate() from a joinable thread's destructor.
  PoolRunner.join();

  std::lock_guard<std::mutex> Lock(g_HandlesMtx);
  ASSERT_EQ(g_CapturedHandles.size(), NThreads)
      << "Each thread should have triggered exactly one write enqueue";

  ur_program_handle_t ExpectedHandle = g_CapturedHandles.front();
  ASSERT_NE(ExpectedHandle, ur_program_handle_t{})
      << "Program handle must be non-null after build completes";

  for (ur_program_handle_t H : g_CapturedHandles)
    EXPECT_EQ(H, ExpectedHandle)
        << "All threads must see the same program handle after the "
           "build-race resolves via waitUntilTransition()";
}

} // namespace
