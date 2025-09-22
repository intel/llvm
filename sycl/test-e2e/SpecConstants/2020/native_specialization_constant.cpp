// native_specialization_constant() returns true only in JIT mode
// on opencl & level-zero backends
// (because only SPIR-V supports specialization constants natively)

// REQUIRES: opencl-aot, ocloc, target-spir

// RUN: %{build} -DJIT -o %t1.out
// RUN: %{run} %t1.out

// RUN: %if any-device-is-gpu %{ %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %s -o %t2.out %}
// RUN: %if gpu %{ %{run} %t2.out %}

// RUN: %if any-device-is-cpu %{ %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 %s -o %t3.out %}
// RUN: %if cpu %{ %{run} %t3.out %}

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/specialization_id.hpp>

constexpr sycl::specialization_id<float> float_id(3.14f);

int main() {
  sycl::queue Q;
  Q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class Kernel>([=](sycl::kernel_handler h) {
      h.get_specialization_constant<float_id>();
    });
  });

#ifdef JIT
  auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Q.get_context());
  assert(bundle.native_specialization_constant());
#else
  auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Q.get_context());
  assert(!bundle.native_specialization_constant());
#endif // JIT

  return 0;
}
