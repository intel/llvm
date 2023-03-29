// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// UNSUPPORTED: cpu || cuda || hip

#include <sycl/sycl.hpp>
#include <vector>
namespace syclex = sycl::ext::oneapi::experimental;

template <size_t ClusterSize> class TestKernel;

template <size_t ClusterSize> void test() {
  sycl::queue Q;

  auto SGSizes = Q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(SGSizes.begin(), SGSizes.end(), 32) == SGSizes.end()) {
    std::cout << "Test skipped due to missing support for sub-group size 32."
              << std::endl;
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

          auto ClusterGroup = syclex::get_cluster_group<ClusterSize>(SG);

          bool Match = true;
          Match &= (ClusterGroup.get_group_id() == (WI / ClusterSize));
          Match &= (ClusterGroup.get_local_id() == (WI % ClusterSize));
          Match &= (ClusterGroup.get_group_range() == (32 / ClusterSize));
          Match &= (ClusterGroup.get_local_range() == ClusterSize);
          MatchAcc[WI] = Match;
          LeaderAcc[WI] = ClusterGroup.leader();
        };
    CGH.parallel_for<TestKernel<ClusterSize>>(NDR, KernelFunc);
  });

  sycl::host_accessor MatchAcc{MatchBuf, sycl::read_only};
  sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_only};
  for (int WI = 0; WI < 32; ++WI) {
    assert(MatchAcc[WI] == true);
    assert(LeaderAcc[WI] == ((WI % ClusterSize) == 0));
  }
}

int main() {
  test<1>();
  test<2>();
  test<4>();
  test<8>();
  test<16>();
  test<32>();
  return 0;
}
