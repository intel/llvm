// RUN: %{build} -fno-sycl-early-optimizations -o %t.out
// RUN: %{run} %t.out
//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/tangle_group.hpp>
#include <vector>
namespace syclex = sycl::ext::oneapi::experimental;

class TestKernel;

int main() {
  sycl::queue Q;

  auto SGSizes = Q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(SGSizes.begin(), SGSizes.end(), 32) == SGSizes.end()) {
    std::cout << "Test skipped due to missing support for sub-group size 32."
              << std::endl;
    return 0;
  }

  // Test for both the full sub-group size and a case with less work than a full
  // sub-group.
  for (size_t WGS : std::array<size_t, 2>{32, 16}) {
    std::cout << "Testing for work size " << WGS << std::endl;

    sycl::buffer<bool, 1> MatchBuf{sycl::range{WGS}};
    sycl::buffer<bool, 1> LeaderBuf{sycl::range{WGS}};

    const auto NDR = sycl::nd_range<1>{WGS, WGS};
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor MatchAcc{MatchBuf, CGH, sycl::write_only};
      sycl::accessor LeaderAcc{LeaderBuf, CGH, sycl::write_only};
      const auto KernelFunc =
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            auto WI = item.get_global_id();
            auto SG = item.get_sub_group();

            // Split into odd and even work-items via control flow.
            // Branches deliberately duplicated to test impact of optimizations.
            // This only reliably works with optimizations disabled right now.
            if (item.get_global_id() % 2 == 0) {
              auto TangleGroup = syclex::get_tangle_group(SG);

              bool Match = true;
              Match &= (TangleGroup.get_group_id() == 0);
              Match &= (TangleGroup.get_local_id() == SG.get_local_id() / 2);
              Match &= (TangleGroup.get_group_range() == 1);
              Match &= (TangleGroup.get_local_range() ==
                        SG.get_local_linear_range() / 2);
              MatchAcc[WI] = Match;
              LeaderAcc[WI] = TangleGroup.leader();
            } else {
              auto TangleGroup = syclex::get_tangle_group(SG);

              bool Match = true;
              Match &= (TangleGroup.get_group_id() == 0);
              Match &= (TangleGroup.get_local_id() == SG.get_local_id() / 2);
              Match &= (TangleGroup.get_group_range() == 1);
              Match &= (TangleGroup.get_local_range() ==
                        SG.get_local_linear_range() / 2);
              MatchAcc[WI] = Match;
              LeaderAcc[WI] = TangleGroup.leader();
            }
          };
      CGH.parallel_for<TestKernel>(NDR, KernelFunc);
    });

    sycl::host_accessor MatchAcc{MatchBuf, sycl::read_only};
    sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_only};
    for (int WI = 0; WI < WGS; ++WI) {
      assert(MatchAcc[WI] == true);
      assert(LeaderAcc[WI] == (WI < 2));
    }
  }
  return 0;
}
