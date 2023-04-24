// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// UNSUPPORTED: cpu || cuda || hip

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

          // Split into odd and even work-items.
          bool Predicate = WI % 2 == 0;
          auto BallotGroup = syclex::get_ballot_group(SG, Predicate);

          // Check function return values match Predicate.
          // NB: Test currently uses exactly one sub-group, but we use SG
          //     below in case this changes in future.
          bool Match = true;
          auto GroupID = (Predicate) ? 1 : 0;
          auto LocalID = SG.get_local_id() / 2;
          Match &= (BallotGroup.get_group_id() == GroupID);
          Match &= (BallotGroup.get_local_id() == LocalID);
          Match &= (BallotGroup.get_group_range() == 2);
          Match &= (BallotGroup.get_local_range() == 16);
          MatchAcc[WI] = Match;
          LeaderAcc[WI] = BallotGroup.leader();
        };
    CGH.parallel_for<TestKernel>(NDR, KernelFunc);
  });

  sycl::host_accessor MatchAcc{MatchBuf, sycl::read_only};
  sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_only};
  for (int WI = 0; WI < 32; ++WI) {
    assert(MatchAcc[WI] == true);
    assert(LeaderAcc[WI] == (WI < 2));
  }
  return 0;
}
