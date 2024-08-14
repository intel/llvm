// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue Q;
  sycl::buffer<int, 3> Buf{sycl::range{32, 32, 3}};

  size_t SubgroupSize = 0;

  {
    sycl::buffer<size_t, 1> SGBuf{&SubgroupSize, 1};
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor WriteAcc{Buf, CGH, sycl::write_only};
      sycl::accessor SGSizeAcc{SGBuf, CGH, sycl::write_only};
      CGH.parallel_for(
          sycl::nd_range<1>(sycl::range{32}, sycl::range{32}),
          [=](sycl::nd_item<1> item) {
            auto SG = item.get_sub_group();

            // Get sub-group size once.
            if (item.get_global_linear_id() == 0)
              SGSizeAcc[0] = SG.get_local_linear_range();

            auto PerWI =
                WriteAcc[SG.get_group_linear_id()][SG.get_local_linear_id()];
            PerWI[0] = SG.leader();
            PerWI[1] = SG.get_group_linear_range();
            PerWI[2] = SG.get_local_linear_range();
          });
    });
  }

  sycl::host_accessor HostAcc{Buf, sycl::read_only};

  const size_t NumSubgroups = 32 / SubgroupSize;
  std::cout << "SubgroupSize " << SubgroupSize << std::endl;
  std::cout << "NumSubgroups " << NumSubgroups << std::endl;

  for (size_t SGNo = 0; SGNo < NumSubgroups; SGNo++) {
    for (size_t WINo = 0; WINo < SubgroupSize; WINo++) {
      const int Leader = WINo == 0 ? 1 : 0;
      assert(HostAcc[SGNo][WINo][0] == Leader);
      assert(HostAcc[SGNo][WINo][1] == NumSubgroups);
      assert(HostAcc[SGNo][WINo][2] == SubgroupSize);
    }
  }

  return 0;
}
