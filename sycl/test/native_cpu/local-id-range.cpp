// REQUIRES: native_cpu_be
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t

#include <clocale>
#include <sycl/sycl.hpp>

int main() {
  const size_t dim = 2;
  using dataT = std::tuple<size_t, size_t, size_t>;
  sycl::range<3> NumOfWorkItems{2 * dim, 2 * (dim + 1), 2 * (dim + 2)};
  sycl::range<3> LocalSizes{dim, dim + 1, dim + 2};
  sycl::buffer<dataT, 3> Buffer(NumOfWorkItems);

  sycl::queue Queue;

  Queue.submit([&](sycl::handler &cgh) {
    sycl::accessor Accessor{Buffer, cgh, sycl::write_only};
    sycl::nd_range<3> TheRange{NumOfWorkItems, LocalSizes};
    cgh.parallel_for<class FillBuffer>(TheRange, [=](sycl::nd_item<3> id) {
      auto localX = id.get_local_id(0);
      auto localY = id.get_local_id(1);
      auto localZ = id.get_local_id(2);

      auto groupX = id.get_group(0);
      auto groupY = id.get_group(1);
      auto groupZ = id.get_group(2);

      auto rangeX = id.get_local_range(0);
      auto rangeY = id.get_local_range(1);
      auto rangeZ = id.get_local_range(2);
      Accessor[groupX * rangeX + localX][groupY * rangeY + localY]
              [groupZ * rangeZ + localZ] = {rangeX, rangeY, rangeZ};
    });
  });

  sycl::host_accessor HostAccessor{Buffer, sycl::read_only};

  for (size_t I = 0; I < NumOfWorkItems[0]; I++) {
    for (size_t J = 0; J < NumOfWorkItems[1]; J++) {
      for (size_t K = 0; K < NumOfWorkItems[2]; K++) {
        auto t = HostAccessor[I][J][K];
        if (std::get<0>(t) != LocalSizes[0] ||
            std::get<1>(t) != LocalSizes[1] ||
            std::get<2>(t) != LocalSizes[2]) {
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
