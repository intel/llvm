// REQUIRES: level_zero

// TODO: The L0 loader on Windows CI does not count zeEventCounterBasedCreate,
// so the matching zeEventDestroy calls are reported as a negative leak there.
// Ignore negative leaks on Windows until the loader is updated.
// DEFINE: %{leak_check_not} = %if windows %{"LEAK = {{[^-]}}"%} %else %{LEAK%}

// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:*" %{l0_leak_check}  %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=%{leak_check_not}

// Tests that additional resources required by USM reductions do not leak.

#include <sycl/detail/core.hpp>

#include <sycl/reduction.hpp>

using namespace sycl;

int main() {
  queue Q;

  nd_range<1> NDRange(range<1>{49 * 5}, range<1>{49});
  std::plus<> BOp;

  int *Out = malloc_shared<int>(1, Q);
  int *In = malloc_shared<int>(49 * 5, Q);
  Q.submit([&](handler &CGH) {
     auto Redu = reduction(Out, 0, BOp);
     CGH.parallel_for<class USMSum>(
         NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
           Sum.combine(In[NDIt.get_global_linear_id()]);
         });
   }).wait();
  sycl::free(In, Q);
  sycl::free(Out, Q);
  return 0;
}
