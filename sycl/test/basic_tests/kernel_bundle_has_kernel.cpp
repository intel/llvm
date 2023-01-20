// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -c -o %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Queue;

  using KernelName = class KernelA;

  Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<KernelName>([=]() {}); });

  auto Bundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::executable>(
          Queue.get_context());

  assert(Bundle.has_kernel<KernelA>());
}
