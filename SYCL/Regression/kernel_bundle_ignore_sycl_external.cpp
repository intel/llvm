// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// XFAIL: cuda || hip_nvidia

#include <sycl/sycl.hpp>

class KernelName;

SYCL_EXTERNAL
int f(int a) { return a + 1; }

int main() {
  const sycl::device Dev{sycl::default_selector{}};
  const sycl::context Ctx{Dev};
  sycl::queue Q{Ctx, Dev};

  assert(sycl::get_kernel_ids().size() == 1);

  sycl::kernel_bundle EmptyKernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev}, {});

  assert(EmptyKernelBundle.get_kernel_ids().size() == 0);

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});
  sycl::kernel_id KernelID = sycl::get_kernel_id<KernelName>();

  assert(KernelBundle.get_kernel_ids().size() == 1);
  assert(KernelBundle.has_kernel(KernelID));

  cl::sycl::buffer<int, 1> Buf(sycl::range<1>{1});
  Q.submit([&](sycl::handler &CGH) {
    auto Acc = Buf.get_access<sycl::access::mode::write>(CGH);
    CGH.use_kernel_bundle(KernelBundle);
    CGH.single_task<KernelName>([=]() { Acc[0] = 42; });
  });
}
