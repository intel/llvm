// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// TODO enable test on CUDA once kernel_bundle is supported
// UNSUPPORTED: cuda

#include <sycl/sycl.hpp>

class TestKernel;

int main() {
  sycl::queue Q;
  sycl::buffer<int, 3> Buf{sycl::range{3, 32, 32}};

  sycl::kernel_id TestKernelID = sycl::get_kernel_id<TestKernel>();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Q.get_context(),
                                                              {TestKernelID});

  Q.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(KernelBundle);
    sycl::accessor Acc{Buf, CGH, sycl::write_only};
    CGH.parallel_for<TestKernel>(
        sycl::nd_range<1>(sycl::range{32}, sycl::range{32}),
        [=](sycl::nd_item<1> item) {
          auto SG = item.get_sub_group();
          Acc[0][SG.get_group_linear_id()][SG.get_local_linear_id()] =
              SG.leader();
          Acc[1][SG.get_group_linear_id()][SG.get_local_linear_id()] =
              SG.get_group_linear_range();
          Acc[2][SG.get_group_linear_id()][SG.get_local_linear_id()] =
              SG.get_local_linear_range();
        });
  });

  sycl::host_accessor Acc{Buf, sycl::read_only};

  sycl::kernel Kernel = KernelBundle.get_kernel(TestKernelID);

  const size_t SubgroupSize =
      Kernel.get_info<sycl::info::kernel_device_specific::max_sub_group_size>(
          Q.get_device(), sycl::range{32, 1, 1});
  const size_t MaxNumSubgroups = 32 / SubgroupSize;

  for (size_t SGNo = 0; SGNo < MaxNumSubgroups; SGNo++) {
    for (size_t WINo = 0; WINo < SubgroupSize; WINo++) {
      const int Leader = WINo == 0 ? 1 : 0;
      assert(Acc[0][SGNo][WINo] == Leader);
      assert(Acc[1][SGNo][WINo] == MaxNumSubgroups);
      assert(Acc[2][SGNo][WINo] == SubgroupSize);
    }
  }

  return 0;
}
