// REQUIRES: aspect-ext_oneapi_ipc_event
// REQUIRES: level_zero_v2_adapter

// DEFINE: %{cpp20} = %if cl_options %{/clang:-std=c++20%} %else %{-std=c++20%}

// RUN: %{build} %level_zero_options -lze_loader -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} %level_zero_options -lze_loader -DUSE_VIEW %{cpp20} -o %t.view.out
// RUN: %{run} %t.view.out

// Full same-process Get / Open / Wait / Put round trip on an IPC event,
// signalled via the Level Zero interop helper. Covers:
//   * the std::vector<std::byte> handle_data_t overload of open() and
//     (with -DUSE_VIEW) the C++20 std::span handle_data_view_t overload,
//   * default-context overloads of put() and open(),
//   * multiple opens of the same handle yielding distinct events,
//   * imported event reports ext_oneapi_ipc_enabled() == false (it cannot
//     be re-exported),
//   * imported event waits successfully because producer was signalled.

#include "Inputs/ipc_event_l0_signal.hpp"
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_event.hpp>

namespace exp = sycl::ext::oneapi::experimental;
namespace ipc = sycl::ext::oneapi::experimental::ipc;

int main() {
  sycl::queue Q;
  sycl::device Dev = Q.get_device();
  sycl::context Ctx = Q.get_context();

  // 1. Create + signal the producer event.
  sycl::event ProducerEvt =
      exp::make_event(Ctx, exp::properties{exp::enable_ipc});
  ipc_event_test::signalEventViaLevelZero(ProducerEvt, Ctx, Dev);

  // 2. Get the IPC handle.
  ipc::handle Handle = ipc::event::get(ProducerEvt);

  // 3. Open in this process via the requested overload.
  sycl::event Imported1, Imported2;
#ifdef USE_VIEW
  ipc::handle_data_view_t View = Handle.data_view();
  if (View.empty())
    return 10;
  Imported1 = ipc::event::open(View, Ctx);
  Imported2 = ipc::event::open(View, Ctx);
#else
  ipc::handle_data_t Bytes = Handle.data();
  if (Bytes.empty())
    return 10;
  Imported1 = ipc::event::open(Bytes, Ctx);
  Imported2 = ipc::event::open(Bytes, Ctx);
#endif

  // 4. Imported events cannot be re-exported, so ext_oneapi_ipc_enabled()
  //    is false.
  if (Imported1.ext_oneapi_ipc_enabled())
    return 1;
  if (Imported2.ext_oneapi_ipc_enabled())
    return 2;

  // 5. They are not the same SYCL handle, even though they came from the
  //    same producer-side IPC bytes (the spec explicitly notes this:
  //    "each call to open may return a unique event object").
  if (Imported1 == Imported2)
    return 3;

  // 6. Wait completes (producer was signalled).
  Imported1.wait();
  Imported2.wait();

  // 7. Default-context put compiles and runs.
  ipc::event::put(Handle, Ctx);

  // Optional: also exercise the default-context open + put overloads.
#ifndef USE_VIEW
  ipc::handle Handle2 = ipc::event::get(ProducerEvt);
  ipc::handle_data_t Bytes2 = Handle2.data();
  sycl::event ImportedDefaultCtx = ipc::event::open(Bytes2);
  ImportedDefaultCtx.wait();
  ipc::event::put(Handle2);
#endif

  return 0;
}
