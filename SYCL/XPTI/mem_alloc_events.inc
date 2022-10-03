#ifdef XPTI_COLLECTOR

#include "Inputs/buffer_info_collector.cpp"

#else

#include <numeric>
#include <sycl/sycl.hpp>
#include <vector>

int main() {
  sycl::queue Q{sycl::default_selector_v};

  sycl::buffer<int, 1> Buf{sycl::range{100}};
  std::vector<int> Vec;
  Vec.resize(100);
  std::iota(Vec.begin(), Vec.end(), 0);

  Q.submit([&](sycl::handler &CGH) {
    sycl::accessor Acc{Buf, CGH, sycl::write_only};
    CGH.copy(Vec.data(), Acc);
  });

  Q.submit([&](sycl::handler &CGH) {
    sycl::accessor Acc{Buf, CGH, sycl::write_only};
    CGH.single_task([=]() { Acc[0] = 42; });
  });

  Q.wait();

  sycl::host_accessor Acc{Buf};
  assert(Acc[0] == 42);

  return 0;
}

#endif
