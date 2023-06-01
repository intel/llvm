// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// REQUIRES: gpu
// UNSUPPORTED: hip

#include <sycl/sycl.hpp>
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

  sycl::buffer<bool, 1> MatchBuf{sycl::range{32}};
  sycl::buffer<bool, 1> LeaderBuf{sycl::range{32}};

  const auto NDR = sycl::nd_range<1>{32, 32};
  Q.submit([&](sycl::handler &CGH) {
    sycl::accessor MatchAcc{MatchBuf, CGH, sycl::write_only};
    sycl::accessor LeaderAcc{LeaderBuf, CGH, sycl::write_only};
    const auto KernelFunc =
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
          auto WI = item.get_global_id();
          auto SG = item.get_sub_group();

          // Due to the unpredictable runtime behavior of opportunistic groups,
          // some values may change from run to run. Check they're in expected
          // ranges and consistent with other groups.
          if (item.get_global_id() % 2 == 0) {
            auto OpportunisticGroup =
                syclex::this_kernel::get_opportunistic_group();

            bool Match = true;
            Match &= (OpportunisticGroup.get_group_id() == 0);
            Match &= (OpportunisticGroup.get_local_id() <
                      OpportunisticGroup.get_local_range());
            Match &= (OpportunisticGroup.get_group_range() == 1);
            Match &= (OpportunisticGroup.get_local_linear_range() <=
                      SG.get_local_linear_range());
            MatchAcc[WI] = Match;
            LeaderAcc[WI] = OpportunisticGroup.leader();
          }
        };
    CGH.parallel_for<TestKernel>(NDR, KernelFunc);
  });

  sycl::host_accessor MatchAcc{MatchBuf, sycl::read_only};
  sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_only};
  uint32_t NumLeaders = 0;
  for (int WI = 0; WI < 32; ++WI) {
    if (WI % 2 == 0) {
      assert(MatchAcc[WI] == true);
      if (LeaderAcc[WI]) {
        NumLeaders++;
      }
    }
  }
  assert(NumLeaders > 0);
  return 0;
}
