// native_specialization_constant() returns true only in JIT mode
// on opencl & level-zero backends
// (because only SPIR-V supports specialization constants natively)

// FIXME: This set is never satisfied all at once in our infrastructure.
// REQUIRES: opencl, level-zero, cpu, gpu, opencl-aot, ocloc

// RUN: %clangxx -fsycl -DJIT %s -o %t.out
// RUN: %{run} %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

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
