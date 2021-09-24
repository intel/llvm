// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Missing __spirv_SubgroupLocalInvocationId on AMD
// XFAIL: hip_amd

#include <sycl/sycl.hpp>

class TestKernel;
class TestKernelCUDA;

int main() {
  sycl::queue Q;
  sycl::buffer<int, 3> Buf{sycl::range{3, 32, 32}};

  size_t SubgroupSize = 0;

  sycl::accessor WriteAcc{Buf, sycl::write_only};
  const auto KernelFunc = [=](sycl::nd_item<1> item) {
    auto SG = item.get_sub_group();
    WriteAcc[0][SG.get_group_linear_id()][SG.get_local_linear_id()] =
        SG.leader();
    WriteAcc[1][SG.get_group_linear_id()][SG.get_local_linear_id()] =
        SG.get_group_linear_range();
    WriteAcc[2][SG.get_group_linear_id()][SG.get_local_linear_id()] =
        SG.get_local_linear_range();
  };

  if (Q.get_backend() != sycl::backend::cuda) {
    sycl::kernel_id TestKernelID = sycl::get_kernel_id<TestKernel>();
    sycl::kernel_bundle KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(Q.get_context(),
                                                                {TestKernelID});

    Q.submit([&](sycl::handler &CGH) {
      CGH.use_kernel_bundle(KernelBundle);
      CGH.require(WriteAcc);
      CGH.parallel_for<TestKernel>(
          sycl::nd_range<1>(sycl::range{32}, sycl::range{32}), KernelFunc);
    });

    sycl::kernel Kernel = KernelBundle.get_kernel(TestKernelID);
    SubgroupSize =
        Kernel.get_info<sycl::info::kernel_device_specific::max_sub_group_size>(
            Q.get_device(), sycl::range{32, 1, 1});
  } else {
    // CUDA sub-group size is 32 by default (size of a warp) so the kernel
    // bundle is not strictly needed to do this test for the CUDA backend.
    // TODO: Remove this special CUDA path once the CUDA backend supports kernel
    // bundles.
    SubgroupSize = 32;
    Q.submit([&](sycl::handler &CGH) {
      CGH.require(WriteAcc);
      CGH.parallel_for<TestKernelCUDA>(
          sycl::nd_range<1>(sycl::range{32}, sycl::range{32}), KernelFunc);
    });
  }

  sycl::host_accessor HostAcc{Buf, sycl::read_only};

  const size_t MaxNumSubgroups = 32 / SubgroupSize;

  for (size_t SGNo = 0; SGNo < MaxNumSubgroups; SGNo++) {
    for (size_t WINo = 0; WINo < SubgroupSize; WINo++) {
      const int Leader = WINo == 0 ? 1 : 0;
      assert(HostAcc[0][SGNo][WINo] == Leader);
      assert(HostAcc[1][SGNo][WINo] == MaxNumSubgroups);
      assert(HostAcc[2][SGNo][WINo] == SubgroupSize);
    }
  }

  return 0;
}
