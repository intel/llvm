// REQUIRES: aspect-ext_oneapi_ipc_event
// REQUIRES: aspect-ext_oneapi_ipc_memory
// REQUIRES: level_zero_v2_adapter
// REQUIRES: arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31
// UNSUPPORTED: windows
// UNSUPPORTED-INTENDED: Cross-process IPC test relies on POSIX semantics.

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E21 (guard): a deferred signal or wait of an IPC event is
// rejected with errc::invalid, in the exporting and in the importing process,
// and leaves the event unchanged and usable through an immediate signal.
//
// The backend event of an IPC event is shared with another process, so a
// signal of it cannot be held back in the scheduler, and a dependency on it
// cannot keep a signal of its own. queue_impl::submit_barrier_direct_impl
// (queue_impl.cpp:629-643 at 8337da70) therefore throws errc::invalid when an
// exported or imported IPC event (hasSharedBackendEvent) would be signaled or
// waited for on the scheduler path, before the signal prepares the event.
//
// Each process, with an in-order queue blocked by a gated host task:
//   1. enqueue_signal_event(Q, IpcEvent)                  -> errc::invalid
//   2. enqueue_wait_event(Q, IpcEvent)                    -> errc::invalid
//   3. enqueue_wait_events(Q, {Completed, IpcEvent})      -> errc::invalid
//   4. enqueue_wait_events(OOOQ, {Deferred, IpcEvent}) on an idle out-of-order
//      queue, where Deferred is a kernel event behind the gate without a
//      backend event                                      -> errc::invalid
//   5. The IPC event's status is unchanged, before and after the gate opens.
// The IPC event has not been signaled before the checks, so the recovery below
// is its first signal (repeated cross-process signals are not used, see
// ipc_event_repeated_signal.cpp).
//
// Recovery: the importing process writes a sentinel into an IPC-shared buffer
// and signals the imported event immediately on its idle in-order queue, then
// waits for it immediately with enqueue_wait_event. The exporting process
// waits for its event immediately with enqueue_wait_event on its idle in-order
// queue and reads the buffer on that queue; the buffer is poisoned first.
//
// clang-format off
// Sentinel protocol:
//   ipcdefer_handles_ready     producer -> consumer  producer's checks done, handles written
//   ipcdefer_signal_done       consumer -> producer  consumer's checks done, signal submitted
//   ipcdefer_consumer_failed   consumer -> producer  a consumer check failed (only on error)
//   ipcdefer_producer_synced   producer -> consumer  producer's wait done
//   ipcdefer_consumer_done     consumer -> producer  consumer closed its handles
// clang-format on

#include "Inputs/ipc_event_sentinel.hpp"
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_event.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <linux/prctl.h>
#include <sys/prctl.h>
#endif

namespace exp = sycl::ext::oneapi::experimental;
namespace ipc = sycl::ext::oneapi::experimental::ipc;
using namespace std::chrono_literals;

static constexpr size_t NumElems = 64;
static constexpr int Sentinel = 0x5A5A5A5A;
static constexpr int Poison = 0x0BADCAFE;
static constexpr auto GateTimeout = 60s;
static constexpr auto WaitTimeout = 60s;

namespace {

struct Gate {
  std::mutex M;
  std::condition_variable CV;
  bool IsOpen = true;
  std::atomic<int> Entered{0};
  std::atomic<bool> TimedOut{false};

  void open() {
    {
      std::lock_guard<std::mutex> L(M);
      IsOpen = true;
    }
    CV.notify_all();
  }
  // Only called while no host task is inside pass().
  void close() {
    std::lock_guard<std::mutex> L(M);
    IsOpen = false;
  }
  void pass() {
    ++Entered;
    std::unique_lock<std::mutex> L(M);
    if (!CV.wait_for(L, GateTimeout, [this] { return IsOpen; }))
      TimedOut = true;
  }
};

Gate QueueGate;

struct GateOpener {
  ~GateOpener() { QueueGate.open(); }
};

[[noreturn]] void fatal(const std::string &Who, const std::string &Msg) {
  std::cerr << "FATAL (" << Who << "): " << Msg << std::endl;
  QueueGate.open();
  std::_Exit(1);
}

template <typename Pred>
bool waitUntil(Pred P, std::chrono::milliseconds Timeout) {
  auto Deadline = std::chrono::steady_clock::now() + Timeout;
  while (!P()) {
    if (std::chrono::steady_clock::now() > Deadline)
      return P();
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

// Bounds a blocking section: exits the process if it is not disarmed in time.
class Watchdog {
public:
  Watchdog(std::string Who, std::string What) {
    T = std::thread([this, Who = std::move(Who), What = std::move(What)]() {
      if (!waitUntil([this] { return Disarmed.load(); }, WaitTimeout))
        fatal(Who, What + " timed out");
    });
  }
  ~Watchdog() {
    Disarmed = true;
    T.join();
  }

private:
  std::atomic<bool> Disarmed{false};
  std::thread T;
};

sycl::info::event_command_status status(const sycl::event &E) {
  return E.get_info<sycl::info::event::command_execution_status>();
}

template <typename Func>
bool expectInvalid(Func &&Call, const std::string &Who, const char *What) {
  try {
    Call();
  } catch (const sycl::exception &Ex) {
    if (Ex.code() == sycl::make_error_code(sycl::errc::invalid))
      return true;
    std::cerr << "FAILED (" << Who << "): " << What
              << ": unexpected exception: " << Ex.what() << std::endl;
    return false;
  }
  std::cerr << "FAILED (" << Who << "): " << What << ": no exception"
            << std::endl;
  return false;
}

// Runs the deferred signal and wait checks of IpcEvent. Returns the number of
// failed checks.
int checkDeferredRejection(sycl::context &Ctx, sycl::device &Dev,
                           sycl::event &IpcEvent, const std::string &Who) {
  GateOpener Opener;
  int Failures = 0;
  sycl::queue Q{Ctx, Dev, {sycl::property::queue::in_order{}}};
  sycl::queue OOOQ{Ctx, Dev};

  // An ordinary completed reusable event, signaled immediately.
  sycl::event Completed = exp::make_event(Ctx);
  exp::enqueue_signal_event(Q, Completed);
  Completed.wait();

  const auto StatusBefore = status(IpcEvent);

  QueueGate.close();
  const int Base = QueueGate.Entered.load();
  Q.submit(
      [&](sycl::handler &CGH) { CGH.host_task([]() { QueueGate.pass(); }); });
  if (!waitUntil([&] { return QueueGate.Entered.load() == Base + 1; },
                 WaitTimeout))
    fatal(Who, "host task did not start");
  // A device event behind the gate, without a backend event.
  sycl::event Deferred = Q.single_task([]() {});

  if (!expectInvalid([&]() { exp::enqueue_signal_event(Q, IpcEvent); }, Who,
                     "deferred signal"))
    ++Failures;
  if (!expectInvalid([&]() { exp::enqueue_wait_event(Q, IpcEvent); }, Who,
                     "deferred wait"))
    ++Failures;
  if (!expectInvalid(
          [&]() {
            exp::enqueue_wait_events(Q, {Completed, IpcEvent});
          },
          Who, "deferred wait, mixed vector"))
    ++Failures;
  if (!expectInvalid(
          [&]() {
            exp::enqueue_wait_events(OOOQ, {Deferred, IpcEvent});
          },
          Who, "wait with a deferred dependency on an idle out-of-order queue"))
    ++Failures;

  if (status(IpcEvent) != StatusBefore) {
    std::cerr << "FAILED (" << Who
              << "): the IPC event's status changed while the queue was "
                 "blocked"
              << std::endl;
    ++Failures;
  }

  QueueGate.open();
  {
    Watchdog W(Who, "draining the checking queues");
    Q.wait_and_throw();
    OOOQ.wait_and_throw();
  }
  if (status(IpcEvent) != StatusBefore) {
    std::cerr << "FAILED (" << Who
              << "): the IPC event's status changed after the queue drained"
              << std::endl;
    ++Failures;
  }
  if (QueueGate.TimedOut.load()) {
    std::cerr << "FAILED (" << Who << "): the gate timed out" << std::endl;
    ++Failures;
  }
  return Failures;
}

} // namespace

static int producer(const std::string &Exe) {
  const std::string Who = "exporting process";
  sycl::queue Q{sycl::property::queue::in_order{}};
  sycl::device Dev = Q.get_device();
  sycl::context Ctx = Q.get_context();

#if defined(__linux__)
  // Allow the unrelated consumer to pidfd_getfd into this process.
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
#endif

  // lit does not clean the working directory between reruns; remove any stale
  // sentinels so waitForFile does not match a leftover file from a prior run.
  ipc_event_test::removeStaleFiles(
      {"ipcdefer_handles_ready", "ipcdefer_signal_done",
       "ipcdefer_consumer_failed", "ipcdefer_producer_synced",
       "ipcdefer_consumer_done"});

  sycl::event Evt =
      exp::make_event(Ctx, exp::properties{exp::enable_ipc{true}});
  if (!Evt.ext_oneapi_ipc_enabled()) {
    std::cerr << "FAILED: make_event(enable_ipc) is not IPC-enabled\n";
    return 1;
  }
  ipc::handle EvtHandle = ipc::event::get(Evt);

  int Rc = checkDeferredRejection(Ctx, Dev, Evt, Who) ? 1 : 0;

  int *Buf = sycl::malloc_device<int>(NumElems, Q);
  Q.fill(Buf, Poison, NumElems).wait();
  ipc::handle MemHandle = ipc::memory::get(Buf, Ctx);

  ipc_event_test::writeHandleFile("ipcdefer_event.bin", EvtHandle);
  ipc_event_test::writeHandleFile("ipcdefer_mem.bin", MemHandle);
  ipc_event_test::touchFile("ipcdefer_handles_ready");

  std::system((Exe + " consumer &").c_str());

  ipc_event_test::waitForFile("ipcdefer_signal_done", 60);
  // The consumer reports a failure before signal_done; its signal may then be
  // missing, so do not wait for it.
  const bool ConsumerFailedEarly =
      std::ifstream{"ipcdefer_consumer_failed"}.good();
  if (ConsumerFailedEarly) {
    std::cerr << "FAILED: the consumer reported a failed check\n";
    ipc_event_test::touchFile("ipcdefer_producer_synced");
    ipc_event_test::waitForFile("ipcdefer_consumer_done", 60);
    ipc::memory::put(MemHandle, Ctx);
    ipc::event::put(EvtHandle, Ctx);
    sycl::free(Buf, Q);
    return 1;
  }

  // Recovery: an immediate wait on the idle in-order queue. The read is
  // ordered after the wait, which is the only thing ordering it after the
  // consumer's write.
  std::vector<int> Host(NumElems);
  {
    Watchdog W(Who, "immediate wait for the consumer's signal");
    exp::enqueue_wait_event(Q, Evt);
    Q.memcpy(Host.data(), Buf, NumElems * sizeof(int)).wait();
    Evt.wait();
  }
  for (size_t I = 0; I < NumElems; ++I) {
    if (Host[I] != Sentinel) {
      std::cerr << "FAILED: Buf[" << I << "] = " << std::hex << Host[I]
                << ", expected " << Sentinel << std::dec << "\n";
      Rc = 1;
      break;
    }
  }
  if (status(Evt) != sycl::info::event_command_status::complete) {
    std::cerr << "FAILED: the event is not complete after the wait\n";
    Rc = 1;
  }

  ipc_event_test::touchFile("ipcdefer_producer_synced");
  ipc_event_test::waitForFile("ipcdefer_consumer_done", 60);
  if (std::ifstream{"ipcdefer_consumer_failed"}.good()) {
    std::cerr << "FAILED: the consumer reported a failed check\n";
    Rc = 1;
  }

  ipc::memory::put(MemHandle, Ctx);
  ipc::event::put(EvtHandle, Ctx);
  sycl::free(Buf, Q);

  if (Rc == 0)
    std::cout << "PASSED: deferred IPC signals and waits rejected\n";
  return Rc;
}

static int consumer() {
  const std::string Who = "importing process";
  sycl::queue Q{sycl::property::queue::in_order{}};
  sycl::device Dev = Q.get_device();
  sycl::context Ctx = Q.get_context();

  ipc_event_test::waitForFile("ipcdefer_handles_ready");
  auto EvtBytes = ipc_event_test::readHandleFile("ipcdefer_event.bin");
  auto MemBytes = ipc_event_test::readHandleFile("ipcdefer_mem.bin");

  sycl::event Imported = ipc::event::open(EvtBytes, Ctx);
  void *Shared = ipc::memory::open(MemBytes, Ctx, Dev);
  int *Buf = static_cast<int *>(Shared);

  int Failures = checkDeferredRejection(Ctx, Dev, Imported, Who);

  // Recovery: an immediate signal on the idle in-order queue, ordered after
  // the sentinel write, then an immediate wait for it.
  Q.parallel_for(sycl::range<1>(NumElems),
                 [=](sycl::id<1> I) { Buf[I] = Sentinel; });
  try {
    exp::enqueue_signal_event(Q, Imported);
  } catch (const sycl::exception &Ex) {
    std::cerr << "FAILED (" << Who << "): immediate signal: " << Ex.what()
              << std::endl;
    ++Failures;
  }
  if (Failures)
    ipc_event_test::touchFile("ipcdefer_consumer_failed");
  ipc_event_test::touchFile("ipcdefer_signal_done");

  if (!Failures) {
    try {
      Watchdog W(Who, "immediate wait for the own signal");
      exp::enqueue_wait_event(Q, Imported);
      Q.wait_and_throw();
    } catch (const sycl::exception &Ex) {
      std::cerr << "FAILED (" << Who << "): immediate wait: " << Ex.what()
                << std::endl;
      ++Failures;
      ipc_event_test::touchFile("ipcdefer_consumer_failed");
    }
  }

  // Stay alive while the producer's wait runs.
  ipc_event_test::waitForFile("ipcdefer_producer_synced", 60);
  ipc::memory::close(Shared, Ctx);
  ipc_event_test::touchFile("ipcdefer_consumer_done");
  return Failures ? 1 : 0;
}

int main(int argc, char *argv[]) {
  GateOpener Opener;
  try {
    if (argc >= 2 && std::string(argv[1]) == "consumer")
      return consumer();
    return producer(argv[0]);
  } catch (const sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    if (argc >= 2 && std::string(argv[1]) == "consumer") {
      ipc_event_test::touchFile("ipcdefer_consumer_failed");
      ipc_event_test::touchFile("ipcdefer_signal_done");
      ipc_event_test::touchFile("ipcdefer_consumer_done");
    }
    return 1;
  }
}
