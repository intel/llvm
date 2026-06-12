// REQUIRES: aspect-ext_oneapi_ipc_event
// REQUIRES: aspect-ext_oneapi_ipc_memory
// REQUIRES: level_zero_v2_adapter
// REQUIRES: arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31
// UNSUPPORTED: windows
// UNSUPPORTED-INTENDED: Cross-process IPC test relies on POSIX semantics.

// RUN: %{build} -lze_loader %level_zero_options -o %t.out
// RUN: %{run} %t.out

// The consumer signals an IPC event and the producer waits on it, with the
// result verified through a shared buffer.
//
// The producer owns the event and an IPC-shared USM buffer. The consumer opens
// both, enqueues a kernel that writes a sentinel into the buffer, and signals
// the event ordered after that kernel without waiting for it. The producer
// waits on the event and reads the buffer back. The event is the only thing
// ordering the write before the read: the "signal submitted" file does not
// imply completion. The buffer is poisoned first, so a wait that returns
// before the work completes reads the wrong value instead of hanging.
//
// clang-format off
// Sentinel protocol:
//   sigdata_handles_ready    producer -> consumer  handles written, safe to open
//   sigdata_signal_done      consumer -> producer  signalling command submitted
//   sigdata_producer_synced  producer -> consumer  producer's wait done
//   sigdata_consumer_done    consumer -> producer  consumer exited cleanly
// clang-format on

#include "Inputs/ipc_event_l0_signal.hpp"
#include "Inputs/ipc_event_sentinel.hpp"
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_event.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
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

// Number of int elements in the shared buffer and the value the consumer
// writes into every element. The poison value pre-fill must differ from it.
static constexpr size_t NumElems = 64;
static constexpr int Sentinel = 0x5A5A5A5A;
static constexpr int Poison = 0x0BADCAFE;

static int producer(const std::string &Exe) {
  sycl::queue Q;
  sycl::device Dev = Q.get_device();
  sycl::context Ctx = Q.get_context();

#if defined(__linux__)
  // Allow the unrelated consumer to pidfd_getfd into this process.
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
#endif

  // lit does not clean the working directory between reruns; remove any stale
  // sentinels so waitForFile does not match a leftover file from a prior run.
  ipc_event_test::removeStaleFiles(
      {"sigdata_handles_ready", "sigdata_signal_done",
       "sigdata_producer_synced", "sigdata_consumer_done"});

  sycl::event Evt = exp::make_event(Ctx, exp::properties{exp::enable_ipc});
  ipc::handle EvtHandle = ipc::event::get(Evt);

  // Allocate and poison the shared buffer before exporting it.
  int *Buf = sycl::malloc_device<int>(NumElems, Q);
  Q.fill(Buf, Poison, NumElems).wait();
  ipc::handle MemHandle = ipc::memory::get(Buf, Ctx);

  ipc_event_test::writeHandleFile("sigdata_event.bin", EvtHandle);
  ipc_event_test::writeHandleFile("sigdata_mem.bin", MemHandle);
  ipc_event_test::touchFile("sigdata_handles_ready");

  std::system((Exe + " consumer &").c_str());

  ipc_event_test::waitForFile("sigdata_signal_done");
  Evt.wait();

  // The wait returned, so the signal (ordered after the consumer's kernel)
  // has completed and the sentinel must be visible now.
  std::vector<int> Host(NumElems);
  Q.memcpy(Host.data(), Buf, NumElems * sizeof(int)).wait();

  int Rc = 0;
  for (size_t I = 0; I < NumElems; ++I) {
    if (Host[I] != Sentinel) {
      std::cerr << "FAILED: Buf[" << I << "] = " << std::hex << Host[I]
                << ", expected " << Sentinel << std::dec << "\n";
      Rc = 1;
      break;
    }
  }

  ipc_event_test::touchFile("sigdata_producer_synced");
  ipc_event_test::waitForFile("sigdata_consumer_done");

  ipc::memory::put(MemHandle, Ctx);
  ipc::event::put(EvtHandle, Ctx);
  sycl::free(Buf, Q);

  if (Rc == 0)
    std::cout << "PASSED: consumer signal carried data\n";
  return Rc;
}

static int consumer() {
  sycl::queue Q;
  sycl::device Dev = Q.get_device();
  sycl::context Ctx = Q.get_context();

  ipc_event_test::waitForFile("sigdata_handles_ready");
  auto EvtBytes = ipc_event_test::readHandleFile("sigdata_event.bin");
  auto MemBytes = ipc_event_test::readHandleFile("sigdata_mem.bin");

  sycl::event Imported = ipc::event::open(EvtBytes, Ctx);
  void *Shared = ipc::memory::open(MemBytes, Ctx, Dev);
  int *Buf = static_cast<int *>(Shared);

  // Enqueue the sentinel write but DO NOT host-synchronize it: the signal must
  // be the only thing that orders this write before the producer's read.
  sycl::event WriteEvt = Q.parallel_for(
      sycl::range<1>(NumElems), [=](sycl::id<1> I) { Buf[I] = Sentinel; });

  // Submit a signal on the imported event that depends on the write kernel,
  // without waiting for it to complete.
  ze_command_list_handle_t SignalList =
      ipc_event_test::submitSignalDependentOnEvent(Imported, WriteEvt, Ctx,
                                                   Dev);
  ipc_event_test::touchFile("sigdata_signal_done");

  // Stay alive while the producer's wait runs.
  ipc_event_test::waitForFile("sigdata_producer_synced");

  // The producer's wait has completed, so the signal (and thus the write) has
  // completed; it is now safe to tear down the command list.
  zeCommandListDestroy(SignalList);
  ipc::memory::close(Shared, Ctx);
  ipc_event_test::touchFile("sigdata_consumer_done");
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc >= 2 && std::string(argv[1]) == "consumer")
    return consumer();
  return producer(argv[0]);
}
