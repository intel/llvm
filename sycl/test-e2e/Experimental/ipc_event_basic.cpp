// REQUIRES: aspect-ext_oneapi_ipc_event
// REQUIRES: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Basic invariants of the SYCL IPC events surface that don't require
// signaling: aspect query, make_event creating an IPC-capable event,
// ext_oneapi_ipc_enabled() reporting correctly on regular vs IPC events.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_event.hpp>

namespace exp = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue Q;
  sycl::device Dev = Q.get_device();
  sycl::context Ctx = Q.get_context();

  // 1. Aspect is true for this device.
  if (!Dev.has(sycl::aspect::ext_oneapi_ipc_event))
    return 1;

  // 2. make_event(enable_ipc) produces an IPC-capable event.
  sycl::event IpcEvt = exp::make_event(Ctx, exp::properties{exp::enable_ipc});
  if (!IpcEvt.ext_oneapi_ipc_enabled())
    return 2;

  // 3. A regular event is NOT IPC-capable.
  sycl::event Plain = Q.submit([&](sycl::handler &H) { H.host_task([] {}); });
  Plain.wait();
  if (Plain.ext_oneapi_ipc_enabled())
    return 3;

  // 4. A default-constructed event is NOT IPC-capable.
  sycl::event Default;
  if (Default.ext_oneapi_ipc_enabled())
    return 4;

  return 0;
}
