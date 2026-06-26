
// REQUIRES: aspect-usm_shared_allocations

// RUN: rm -rf %t.dir; mkdir -p %t.dir

// DEFINE: %{dynamic_lib_suffix} = %if windows %{dll%} %else %{so%}
// RUN: %clangxx -fsycl %{sycl_target_opts} %fPIC %shared_lib %S/Inputs/a_kernel_id.cpp %S/Inputs/a_kernel.cpp -o %t.dir/liba.%{dynamic_lib_suffix}
// RUN: %clangxx -fsycl %{sycl_target_opts} %fPIC %shared_lib %S/Inputs/b_kernel_id.cpp %S/Inputs/b_kernel.cpp -o %t.dir/libb.%{dynamic_lib_suffix}
// RUN: %if !windows %{%{run-aux}%} \
// RUN: %clangxx -fsycl %{sycl_target_opts} %s -o %t.dir/%{t:stem}.out -L%t.dir \
// RUN: %if windows                                                                    \
// RUN:   %{%t.dir/liba.lib %t.dir/libb.lib%}                                                     \
// RUN: %else                                                                          \
// RUN:   %{-L%t.dir -la -lb -Wl,-rpath=%t.dir%}

// RUN: %{run} %t.dir/%{t:stem}.out
#include <sycl/usm.hpp>

#include "Inputs/a.hpp"
#include "Inputs/b.hpp"
#include "Inputs/common.hpp"

using namespace sycl;

// This test simply captures the current way we handle duplicate kernel names
// in multiiple DSOs: there's no diagnostic issued for this, and the backend
// will pick one of the two images, which should remain the same no matter
// how we attempt to enqueue any of the two kernels.

int main() {
  queue Q;
  int *Ptr = malloc_shared<int>(1, Q);

  enqueueWithKernelId<TestKernel>(Q, Ptr);
  Q.wait();
  int FirstResult = *Ptr;

  auto RunTest = [&](auto Func) {
    *Ptr = 0;
    Func(Q, Ptr);
    Q.wait();
    assert(*Ptr == FirstResult);
  };

  RunTest(submitKernelDirectA);
  RunTest(submitKernelWithIdA);
  RunTest(submitKernelDirectB);
  RunTest(submitKernelWithIdB);

  free(Ptr, Q);
  return 0;
}