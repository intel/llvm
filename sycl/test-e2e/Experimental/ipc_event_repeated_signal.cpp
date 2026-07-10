// REQUIRES: aspect-ext_oneapi_ipc_event
// REQUIRES: aspect-ext_oneapi_ipc_memory
// REQUIRES: level_zero_v2_adapter
// REQUIRES: arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31
// UNSUPPORTED: windows
// UNSUPPORTED-INTENDED: Cross-process IPC test relies on POSIX semantics.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// A handle is opened once and waited on across two separate producer signals
// without reopening, with each wait verified through a shared buffer.
//
// The producer owns the event and an IPC-shared USM buffer. For each round it
// writes a distinct value into the buffer with a kernel and signals the event
// ordered after that kernel without waiting for it. The consumer opens the
// event and the buffer once, then per round waits and reads the buffer back.
// The event is the only thing ordering each round's write before the read, so
// a wait that returns before the work completes reads the wrong value instead
// of hanging. The rounds are serialized with a handshake so the producer does
// not overwrite round #1's value before the consumer reads it.
//
// clang-format off
// Sentinel protocol:
//   repeat_handles_ready    producer -> consumer  handles written, safe to open
//   repeat_signal1_ready    producer -> consumer  signal #1 command submitted
//   repeat_consumed1        consumer -> producer  value #1 read, safe to reuse
//   repeat_signal2_ready    producer -> consumer  signal #2 command submitted
//   repeat_consumed2        consumer -> producer  both rounds done (verdict ready)
//   repeat_consumer_failed  consumer -> producer  a data check failed (only on error)
//   repeat_producer_done    producer -> consumer  verdict read, consumer may close
//   repeat_consumer_done    consumer -> producer  handles closed, safe to release
// clang-format on
//
// The consumer runs in the background so its exit code is not observable by the
// producer; it reports a failed data check by creating repeat_consumer_failed
// before signaling repeat_consumed2, which the producer checks for.

#include "Inputs/ipc_event_sentinel.hpp"
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_event.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

#if defined(__linux__)
#include <linux/prctl.h>
#include <sys/prctl.h>
#endif

namespace exp = sycl::ext::oneapi::experimental;
namespace ipc = sycl::ext::oneapi::experimental::ipc;

static constexpr size_t NumElems = 64;
static constexpr int Value1 = 0x11111111;
static constexpr int Value2 = 0x22222222;
static constexpr int Poison = 0x0BADCAFE;

static int producer(const std::string &Exe) {
  // In-order queue so each signal is ordered after its write kernel.
  sycl::queue Q{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q.get_context();

#if defined(__linux__)
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
#endif

  // lit does not clean the working directory between reruns; remove any stale
  // sentinels so waitForFile does not match a leftover file from a prior run.
  ipc_event_test::removeStaleFiles(
      {"repeat_handles_ready", "repeat_signal1_ready", "repeat_consumed1",
       "repeat_signal2_ready", "repeat_consumed2", "repeat_consumer_failed",
       "repeat_producer_done", "repeat_consumer_done"});

  sycl::event Evt = exp::make_event(Ctx, exp::properties{exp::enable_ipc});
  ipc::handle EvtHandle = ipc::event::get(Evt);

  int *Buf = sycl::malloc_device<int>(NumElems, Q);
  Q.fill(Buf, Poison, NumElems).wait();
  ipc::handle MemHandle = ipc::memory::get(Buf, Ctx);

  ipc_event_test::writeHandleFile("repeat_event.bin", EvtHandle);
  ipc_event_test::writeHandleFile("repeat_mem.bin", MemHandle);
  ipc_event_test::touchFile("repeat_handles_ready");

  std::system((Exe + " consumer &").c_str());

  // Round 1: write Value1, then signal the event (ordered after the write by
  // the in-order queue) without waiting.
  Q.parallel_for(sycl::range<1>(NumElems),
                 [=](sycl::id<1> I) { Buf[I] = Value1; });
  exp::enqueue_signal_event(Q, Evt);
  ipc_event_test::touchFile("repeat_signal1_ready");

  // Wait until the consumer has read Value1 before overwriting the buffer.
  ipc_event_test::waitForFile("repeat_consumed1");

  // Round 2: reuse the same event for a second write.
  Q.parallel_for(sycl::range<1>(NumElems),
                 [=](sycl::id<1> I) { Buf[I] = Value2; });
  exp::enqueue_signal_event(Q, Evt);
  ipc_event_test::touchFile("repeat_signal2_ready");

  ipc_event_test::waitForFile("repeat_consumed2");

  // The consumer signals a failed data check out-of-band, since its exit code
  // is not observable (it runs in the background).
  bool ConsumerFailed = std::ifstream{"repeat_consumer_failed"}.good();

  // Let the consumer close its imported handles before we release the backing
  // event and allocation, otherwise its close races with our free.
  ipc_event_test::touchFile("repeat_producer_done");
  ipc_event_test::waitForFile("repeat_consumer_done");

  ipc::memory::put(MemHandle, Ctx);
  ipc::event::put(EvtHandle, Ctx);
  sycl::free(Buf, Q);

  if (ConsumerFailed) {
    std::cerr << "FAILED: consumer reported a data-check failure\n";
    return 1;
  }
  std::cout << "PASSED: repeated signal carried data\n";
  return 0;
}

static int consumer() {
  sycl::queue Q;
  sycl::device Dev = Q.get_device();
  sycl::context Ctx = Q.get_context();

  ipc_event_test::waitForFile("repeat_handles_ready");
  auto EvtBytes = ipc_event_test::readHandleFile("repeat_event.bin");
  auto MemBytes = ipc_event_test::readHandleFile("repeat_mem.bin");

  // Open once, wait twice.
  sycl::event Imported = ipc::event::open(EvtBytes, Ctx);
  void *Shared = ipc::memory::open(MemBytes, Ctx, Dev);
  int *Buf = static_cast<int *>(Shared);

  auto checkValue = [&](int Expected, const char *Round) -> bool {
    std::vector<int> Host(NumElems);
    Q.memcpy(Host.data(), Buf, NumElems * sizeof(int)).wait();
    for (size_t I = 0; I < NumElems; ++I) {
      if (Host[I] != Expected) {
        std::cerr << "FAILED (" << Round << "): Buf[" << I << "] = " << std::hex
                  << Host[I] << ", expected " << Expected << std::dec << "\n";
        return false;
      }
    }
    return true;
  };

  int Rc = 0;

  ipc_event_test::waitForFile("repeat_signal1_ready");
  Imported.wait();
  if (!checkValue(Value1, "signal #1"))
    Rc = 1;
  ipc_event_test::touchFile("repeat_consumed1");

  if (Rc == 0) {
    ipc_event_test::waitForFile("repeat_signal2_ready");
    Imported.wait();
    if (!checkValue(Value2, "signal #2"))
      Rc = 1;
  }

  // Report a failed data check out-of-band before releasing the producer, so
  // its verdict survives this background process's discarded exit code.
  if (Rc != 0)
    ipc_event_test::touchFile("repeat_consumer_failed");
  ipc_event_test::touchFile("repeat_consumed2");

  ipc_event_test::waitForFile("repeat_producer_done");
  ipc::memory::close(Shared, Ctx);
  ipc_event_test::touchFile("repeat_consumer_done");
  return Rc;
}

int main(int argc, char *argv[]) {
  if (argc >= 2 && std::string(argv[1]) == "consumer")
    return consumer();
  return producer(argv[0]);
}
