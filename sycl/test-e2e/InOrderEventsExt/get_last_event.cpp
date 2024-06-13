// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the ext_oneapi_get_last_event extension member on in-order queues.
// NOTE: The extension does not guarantee that the SYCL events returned by this
//       extension API are equal to the ones returned by the latest submission,
//       only that the underlying native events are. Currently DPC++ implements
//       this in a way that guarantees it, but this can change in the future.
//       If it changes then so should this test.
// OBS: The above note does not apply to equality of events returned after a
//      call to ext_oneapi_set_external_event.

#include <sycl/detail/core.hpp>
#include <sycl/detail/host_task_impl.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <iostream>

template <typename F>
int Check(const sycl::queue &Q, const char *CheckName, const F &CheckFunc) {
  sycl::event E = CheckFunc();
  if (E != Q.ext_oneapi_get_last_event()) {
    std::cout << "Failed " << CheckName << std::endl;
    return 1;
  }
  return 0;
}

int main() {
  sycl::queue Q{{sycl::property::queue::in_order{}}};

  int Failed = 0;

  Failed += Check(Q, "single_task", [&]() { return Q.single_task([]() {}); });

  Failed += Check(Q, "parallel_for",
                  [&]() { return Q.parallel_for(32, [](sycl::id<1>) {}); });

  Failed += Check(Q, "host_task", [&]() {
    return Q.submit([&](sycl::handler &CGH) { CGH.host_task([]() {}); });
  });

  // For external event, the equality of events is guaranteed by the extension.
  sycl::event ExternalEvent = Q.single_task([]() {});
  Failed += Check(Q, "ext_oneapi_set_external_event", [&]() {
    Q.ext_oneapi_set_external_event(ExternalEvent);
    return ExternalEvent;
  });

  if (!Q.get_device().has(sycl::aspect::usm_shared_allocations))
    return Failed;
  constexpr size_t N = 64;
  int *Data1 = sycl::malloc_shared<int>(N, Q);
  int *Data2 = sycl::malloc_shared<int>(N, Q);

  Failed += Check(Q, "fill", [&]() { return Q.fill<int>(Data1, 0, N); });

  Failed +=
      Check(Q, "memset", [&]() { return Q.memset(Data1, 0, N * sizeof(int)); });

  Failed += Check(Q, "memcpy",
                  [&]() { return Q.memcpy(Data1, Data2, N * sizeof(int)); });

  Failed += Check(Q, "copy", [&]() { return Q.memcpy(Data1, Data2, N); });

  Q.wait_and_throw();

  sycl::free(Data1, Q);
  sycl::free(Data2, Q);

  return Failed;
}
