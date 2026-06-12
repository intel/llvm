// REQUIRES: aspect-ext_oneapi_ipc_event
// REQUIRES: level_zero_v2_adapter

// RUN: %{build} %level_zero_options -lze_loader -o %t.out
// RUN: %{run} %t.out

// Lifetime invariants of the IPC events surface:
//   * put() on the producer-side handle does not invalidate the producer
//     event; a fresh get() on the same producer still yields a usable handle.
//   * An imported event remains usable after the producer-side event is
//     destroyed, as long as the imported event is still alive.

#include "Inputs/ipc_event_l0_signal.hpp"
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_event.hpp>

#include <memory>

namespace exp = sycl::ext::oneapi::experimental;
namespace ipc = sycl::ext::oneapi::experimental::ipc;

int main() {
  sycl::queue Q;
  sycl::device Dev = Q.get_device();
  sycl::context Ctx = Q.get_context();

  // 1. put does not invalidate the producer event.
  {
    sycl::event ProducerEvt =
        exp::make_event(Ctx, exp::properties{exp::enable_ipc});
    ipc_event_test::signalEventViaLevelZero(ProducerEvt, Ctx, Dev);

    ipc::handle H1 = ipc::event::get(ProducerEvt);
    ipc::event::put(H1, Ctx);

    // Producer is still usable: fresh get must succeed.
    ipc::handle H2 = ipc::event::get(ProducerEvt);
    if (H2.data().empty())
      return 1;

    sycl::event Imp = ipc::event::open(H2.data(), Ctx);
    Imp.wait();
    ipc::event::put(H2, Ctx);
  }

  // 2. Imported event survives release of the producer.
  {
    // Wrap the producer event in unique_ptr so we can drop it explicitly.
    auto Producer = std::make_unique<sycl::event>(
        exp::make_event(Ctx, exp::properties{exp::enable_ipc}));
    ipc_event_test::signalEventViaLevelZero(*Producer, Ctx, Dev);

    ipc::handle H = ipc::event::get(*Producer);
    sycl::event Imported = ipc::event::open(H.data(), Ctx);

    // Drop the producer event; the imported event must remain usable.
    Producer.reset();

    Imported.wait();
    ipc::event::put(H, Ctx);
  }

  return 0;
}
