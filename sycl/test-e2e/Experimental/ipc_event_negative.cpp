// REQUIRES: aspect-ext_oneapi_ipc_event
// REQUIRES: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Negative path coverage of the SYCL IPC events surface:
//   * ipc::event::get on a non-IPC event throws errc::invalid.
//   * ipc::event::get on an imported event throws errc::invalid
//     (imported events cannot be re-exported).
//   * ipc::event::open with a buffer of the wrong size throws
//     errc::invalid.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ipc_event.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

namespace exp = sycl::ext::oneapi::experimental;
namespace ipc = sycl::ext::oneapi::experimental::ipc;

static int expectInvalid(const sycl::exception &E) {
  return E.code() == sycl::make_error_code(sycl::errc::invalid) ? 0 : 1;
}

int main() {
  sycl::queue Q;
  sycl::context Ctx = Q.get_context();

  // 1. get on a non-IPC event -> errc::invalid.
  {
    sycl::event Plain = Q.submit([&](sycl::handler &H) { H.host_task([] {}); });
    Plain.wait();
    try {
      (void)ipc::event::get(Plain);
      return 1;
    } catch (const sycl::exception &E) {
      if (expectInvalid(E))
        return 2;
    }
  }

  // 2. get on an imported event -> errc::invalid (cannot be re-exported).
  {
    sycl::event Producer =
        exp::make_event(Ctx, exp::properties{exp::enable_ipc});
    exp::enqueue_signal_event(Q, Producer);
    Producer.wait();
    ipc::handle H = ipc::event::get(Producer);
    sycl::event Imported = ipc::event::open(H.data(), Ctx);
    try {
      (void)ipc::event::get(Imported);
      ipc::event::put(H, Ctx);
      return 7;
    } catch (const sycl::exception &E) {
      ipc::event::put(H, Ctx);
      if (expectInvalid(E))
        return 8;
    }
  }

  // 3. open with a wrong-size buffer -> errc::invalid.
  {
    ipc::handle_data_t Bogus(7, std::byte{0x42});
    try {
      (void)ipc::event::open(Bogus, Ctx);
      return 3;
    } catch (const sycl::exception &E) {
      if (expectInvalid(E))
        return 4;
    }
  }

  return 0;
}
