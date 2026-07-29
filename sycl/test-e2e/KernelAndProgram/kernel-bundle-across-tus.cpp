// RUN: %{build} -DMAIN -c -o %t.main.o
// RUN: %{build} -c -o %t.kernel.o
// RUN: %clangxx -fsycl %{sycl_target_opts} %t.main.o %t.kernel.o -o %t.out
// RUN: %{run} %t.out

// DEFINE: %{dynamic_lib_suffix} = %if windows %{dll%} %else %{so%}
// RUN: rm -rf %t.dir; mkdir -p %t.dir
// RUN: %clangxx -fsycl %{sycl_target_opts} %fPIC %shared_lib %s -o %t.dir/libkernel.%{dynamic_lib_suffix}
// RUN: %if !windows %{%{run-aux}%} \
// RUN: %clangxx -fsycl %{sycl_target_opts} %t.main.o -o %t.dir/%{t:stem}.out -L%t.dir \
// RUN: %if windows                                                                    \
// RUN:   %{%t.dir/libkernel.lib%}                                                     \
// RUN: %else                                                                          \
// RUN:   %{-L%t.dir -lkernel -Wl,-rpath=%t.dir%}

// RUN: %{run} %t.dir/%{t:stem}.out
#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#if defined(_WIN32)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT
#endif

class KernelA;
API_EXPORT void submitKernel(sycl::queue &Q);
#ifndef MAIN
void submitKernel(sycl::queue &Q) {
  Q.submit([&](sycl::handler &CGH) { CGH.single_task<KernelA>([=]() {}); });
}
#else
int main() {
  sycl::queue Queue;
  sycl::device Dev = Queue.get_device();
  sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  auto FoundKernel = KernelBundle.get_kernel<KernelA>();
  auto ExpectedKernel = KernelBundle.get_kernel(sycl::get_kernel_id<KernelA>());

  // operator== isn't guaranteed to work in this scenario, so compare traits
  // about the kernels instead.
  auto FoundKernelName =
      FoundKernel.get_info<sycl::info::kernel::function_name>();
  auto ExpectedKernelName =
      ExpectedKernel.get_info<sycl::info::kernel::function_name>();

  auto FoundKernelContext = FoundKernel.get_info<sycl::info::kernel::context>();
  auto ExpectedKernelContext =
      ExpectedKernel.get_info<sycl::info::kernel::context>();

  assert(FoundKernelName == ExpectedKernelName);
  assert(FoundKernelContext == ExpectedKernelContext);

  return 0;
}
#endif
