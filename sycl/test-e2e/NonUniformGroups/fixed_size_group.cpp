// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// REQUIRES: gpu
// UNSUPPORTED: hip
// REQUIRES: sg-32

#include <sycl/sycl.hpp>
#include <vector>
namespace syclex = sycl::ext::oneapi::experimental;

template <size_t PartitionSize> class TestKernel;

template <size_t PartitionSize> void test() {
  sycl::queue Q;

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

          auto Partition = syclex::get_fixed_size_group<PartitionSize>(SG);

          bool Match = true;
          Match &= (Partition.get_group_id() == (WI / PartitionSize));
          Match &= (Partition.get_local_id() == (WI % PartitionSize));
          Match &= (Partition.get_group_range() == (32 / PartitionSize));
          Match &= (Partition.get_local_range() == PartitionSize);
          MatchAcc[WI] = Match;
          LeaderAcc[WI] = Partition.leader();
        };
    CGH.parallel_for<TestKernel<PartitionSize>>(NDR, KernelFunc);
  });

  sycl::host_accessor MatchAcc{MatchBuf, sycl::read_only};
  sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_only};
  for (int WI = 0; WI < 32; ++WI) {
    assert(MatchAcc[WI] == true);
    assert(LeaderAcc[WI] == ((WI % PartitionSize) == 0));
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
