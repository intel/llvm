// REQUIRES: aspect-ext_oneapi_ipc_event
// REQUIRES: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Check that get() can be called multiple times on the same producer event;
// each call returns an independent usable handle.

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

    ipc::handle_data_t H1 = ipc::event::get(ProducerEvt);

    // Producer is still usable: fresh get must succeed.
    ipc::handle_data_t H2 = ipc::event::get(ProducerEvt);
    if (H2.empty())
      return 1;

    sycl::event Imp = ipc::event::open(H2, Ctx);
    Imp.wait();
  }

  return 0;
}
