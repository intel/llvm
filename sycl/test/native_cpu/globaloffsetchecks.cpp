// REQUIRES: native_cpu
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR=native_cpu:cpu %t

#include <sycl/sycl.hpp>

#include <cassert>
#include <iostream>
#include <numeric>

using namespace sycl;

template <int dim, class AT> auto getRangeKernel(AT B) {
  return [=](item<dim> index) {
    B[index.get_id()] =
        int3{index.get_id()[0], index.get_range()[0], index.get_offset()[0]};
  };
}

template <int dim, class AT> auto getNDRangeKernel(AT B) {
  return [=](nd_item<dim> index) {
    B[index.get_global_id()] =
        int3{index.get_global_id()[0], index.get_global_range()[0],
             index.get_offset()[0]};
  };
}

template <int dim, int Range, int LRange, int Offset> int testRange() {

  std::vector<int3> data(Range + Offset + 4, int3{-1});
  const range<dim> globalRange(Range);
  const id<dim> globalOffset(Offset);
  const range<dim> localRange(LRange);
  const nd_range<dim> ndRange(globalRange, localRange, globalOffset);
  {
    buffer<int3, dim> b(data.data(), range<dim>(10),
                        {property::buffer::use_host_ptr()});
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto B = b.template get_access<access::mode::read_write>(cgh);
      if constexpr (LRange > 0)
        cgh.parallel_for(ndRange, getNDRangeKernel<dim>(B));
      else
        cgh.parallel_for(globalRange, globalOffset, getRangeKernel<dim>(B));
    });
  }
  for (int i = 0; i < data.size(); i++) {
    const int id = data[i].s0();
    const int range = data[i].s1();
    const int offset = data[i].s2();
    if (i < globalOffset[0] || i >= globalOffset[0] + Range) {
      if (id == -1 && range == -1 && offset == -1)
        continue;
    } else {
      if (id == i && range == globalRange[0] && offset == globalOffset[0])
        continue;
    }
    return 100 + i;
  }
  return 0;
}

// todo: different dimensions
int main() {
  if (int r = testRange<1 /*Dimension*/, 10 /*range*/, 0 /*Range*/,
                        4 /*global offset*/>())
    return r;
  if (int r = testRange<1 /*Dimension*/, 10 /*range*/, 0 /*Range*/,
                        0 /*global offset*/>())
    return r;
  if (int r = testRange<1 /*Dimension*/, 10 /*range*/, 2 /*LRange*/,
                        4 /*global offset*/>())
    return r;
  if (int r = testRange<1 /*Dimension*/, 10 /*range*/, 2 /*LRange*/,
                        0 /*global offset*/>())
    return r;
  if (int r = testRange<1 /*Dimension*/, 12 /*range*/, 3 /*LRange*/,
                        9 /*global offset*/>())
    return r;
  return 0;
}
