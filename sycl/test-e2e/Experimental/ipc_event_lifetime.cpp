// REQUIRES: aspect-ext_oneapi_ipc_event
// REQUIRES: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Check that put() on the producer-side handle does not invalidate the producer
// event; a fresh get() on the same producer still yields a usable handle.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_event.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

#include <memory>

namespace exp = sycl::ext::oneapi::experimental;
namespace ipc = sycl::ext::oneapi::experimental::ipc;

int main() {
  sycl::queue Q;
  sycl::context Ctx = Q.get_context();

  {
    sycl::event ProducerEvt =
        exp::make_event(Ctx, exp::properties{exp::enable_ipc});
    exp::enqueue_signal_event(Q, ProducerEvt);
    ProducerEvt.wait();

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

  return 0;
}
