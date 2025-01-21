// REQUIRES: native_cpu
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t

#include <sycl/sycl.hpp>

int main() {
  const size_t dim = 4;
  using dataT = std::tuple<size_t, size_t, size_t>;
  sycl::range<3> NumOfWorkItems{dim, dim + 1, dim + 2};
  sycl::buffer<dataT, 3> Buffer(NumOfWorkItems);

  sycl::queue Queue;

  // Basic test for get_global_id and get_global_range,
  // we perform acc[id] = range on 3 dimensions.
  Queue.submit([&](sycl::handler &cgh) {
    sycl::accessor Accessor{Buffer, cgh, sycl::write_only};
    cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](sycl::item<3> WIid) {
      Accessor[WIid[0]][WIid[1]][WIid[2]] = {
          WIid.get_range(0), WIid.get_range(1), WIid.get_range(2)};
    });
  });

  sycl::host_accessor HostAccessor{Buffer, sycl::read_only};

  for (size_t I = 0; I < NumOfWorkItems[0]; I++) {
    for (size_t J = 0; J < NumOfWorkItems[1]; J++) {
      for (size_t K = 0; K < NumOfWorkItems[2]; K++) {
        auto t = HostAccessor[I][J][K];
        if (std::get<0>(t) != NumOfWorkItems[0] ||
            std::get<1>(t) != NumOfWorkItems[1] ||
            std::get<2>(t) != NumOfWorkItems[2]) {
          std::cout << "Mismatch at element " << I << ", " << J << ", " << K
                    << "\n";
          return 1;
        }
      }
    }
  }

  std::cout << "The results are correct!" << std::endl;

  return 0;
}
