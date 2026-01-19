// RUN: %clangxx -fsyntax-only -fsycl %s

// Verify that we can pass top-level special type parameters to free function
// kernels.

#include <sycl/sycl.hpp>

using namespace sycl;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void foo(accessor<int, 1> acc, local_accessor<int, 1> lacc, sampler S,
         ext::oneapi::experimental::annotated_ptr<int> ptr) {}

int main() {
  queue Q;
  kernel_bundle bundle =
      get_kernel_bundle<bundle_state::executable>(Q.get_context());
  kernel_id id = ext::oneapi::experimental::get_kernel_id<foo>();
  kernel Kernel = bundle.get_kernel(id);
  Q.submit([&](handler &h) {
    accessor<int, 1> acc;
    local_accessor<int, 1> lacc;
    sycl::sampler S(sycl::coordinate_normalization_mode::unnormalized,
                    sycl::addressing_mode::clamp,
                    sycl::filtering_mode::nearest);
    ext::oneapi::experimental::annotated_ptr<int> ptr;
    sycl::stream str(8192, 1024, h);
    h.set_args(acc, lacc, S, str, ptr);
    h.parallel_for(nd_range{{1}, {1}}, Kernel);
  });
  return 0;
}
