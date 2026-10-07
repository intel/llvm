// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/oneapi-src/level-zero/issues/512

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests async_malloc, async_malloc_from_pool and async_free when they are
// called on a queue or on a handler, in both cases without requesting an event.
// Each scenario allocates, fills the allocation from a kernel, copies it back
// and frees it.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>

#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

constexpr size_t Width = 8;

// Fills the allocation with the global ids and copies it back to Out. The copy
// depends on the kernel explicitly, as an out-of-order queue does not order the
// two.
template <typename KernelName>
void fillAndCopyBack(sycl::queue &Q, void *Alloc, std::vector<char> &Out) {
  sycl::event Fill =
      Q.parallel_for<KernelName>(sycl::range<1>{Width}, [=](sycl::id<1> Id) {
        static_cast<char *>(Alloc)[Id] = static_cast<char>(Id);
      });
  Q.memcpy(Out.data(), Alloc, Width, Fill);
}

bool validate(const std::vector<char> &Out, const char *Name) {
  for (size_t I = 0; I < Width; ++I) {
    if (Out[I] != static_cast<char>(I)) {
      std::cerr << Name << ": result mismatch at " << I << "! Expected: " << I
                << ", actual: " << static_cast<int>(Out[I]) << std::endl;
      return false;
    }
  }
  return true;
}

class InOrderKernel;
class InOrderPoolKernel;
class HostTaskKernel;
class OutOfOrderKernel;
class HandlerKernel;

int main() {
  bool Pass = true;

  {
    // In-order queue: the queue itself orders everything.
    sycl::queue Q{sycl::property::queue::in_order{}};
    std::vector<char> Out(Width, 0);

    void *Alloc = syclexp::async_malloc(Q, sycl::usm::alloc::device, Width);
    fillAndCopyBack<InOrderKernel>(Q, Alloc, Out);
    syclexp::async_free(Q, Alloc);
    Q.wait_and_throw();

    Pass &= validate(Out, "in-order");
  }

  {
    // Allocating from an explicit memory pool, repeatedly.
    sycl::queue Q{sycl::property::queue::in_order{}};
    syclexp::memory_pool Pool{Q.get_context(), Q.get_device(),
                              sycl::usm::alloc::device};
    std::vector<char> Out(Width, 0);

    // Allocate and free repeatedly to make sure that no state is accumulated
    // between the submissions.
    for (int I = 0; I < 4; ++I) {
      void *Alloc = syclexp::async_malloc_from_pool(Q, Width, Pool);
      fillAndCopyBack<InOrderPoolKernel>(Q, Alloc, Out);
      syclexp::async_free(Q, Alloc);
    }
    Q.wait_and_throw();

    Pass &= validate(Out, "in-order pool");
  }

  {
    // A host task cannot be ordered by the backend, so the commands after it
    // have to go through the scheduler.
    sycl::queue Q{sycl::property::queue::in_order{}};
    std::vector<char> Out(Width, 0);
    bool HostTaskExecuted = false;

    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([&]() { HostTaskExecuted = true; });
    });

    void *Alloc = syclexp::async_malloc(Q, sycl::usm::alloc::device, Width);
    fillAndCopyBack<HostTaskKernel>(Q, Alloc, Out);
    syclexp::async_free(Q, Alloc);
    Q.wait_and_throw();

    if (!HostTaskExecuted) {
      std::cerr << "host task: not executed!" << std::endl;
      Pass = false;
    }
    Pass &= validate(Out, "host task");
  }

  {
    // Out-of-order queue: nothing is ordered implicitly, so the ordering is
    // requested with barriers.
    sycl::queue Q;
    std::vector<char> Out(Width, 0);

    void *Alloc = syclexp::async_malloc(Q, sycl::usm::alloc::device, Width);
    Q.ext_oneapi_submit_barrier();
    fillAndCopyBack<OutOfOrderKernel>(Q, Alloc, Out);
    Q.ext_oneapi_submit_barrier();
    syclexp::async_free(Q, Alloc);
    Q.wait_and_throw();

    Pass &= validate(Out, "out-of-order");
  }

  {
    // The handler overloads submitted without requesting an event must not
    // create, and thereby leak, an event either.
    sycl::queue Q{sycl::property::queue::in_order{}};
    std::vector<char> Out(Width, 0);

    for (int I = 0; I < 4; ++I) {
      void *Alloc = nullptr;
      syclexp::submit(Q, [&](sycl::handler &CGH) {
        Alloc = syclexp::async_malloc(CGH, sycl::usm::alloc::device, Width);
      });
      fillAndCopyBack<HandlerKernel>(Q, Alloc, Out);
      syclexp::submit(
          Q, [&](sycl::handler &CGH) { syclexp::async_free(CGH, Alloc); });
    }
    Q.wait_and_throw();

    Pass &= validate(Out, "handler");
  }

  if (!Pass) {
    std::cerr << "Test failed!" << std::endl;
    return 1;
  }

  std::cout << "Test passed!" << std::endl;
  return 0;
}
