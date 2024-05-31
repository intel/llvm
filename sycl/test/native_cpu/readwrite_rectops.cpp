// REQUIRES: native_cpu
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR=native_cpu:cpu %t

#include <sycl/sycl.hpp>

#include <cassert>
#include <numeric>

using namespace sycl;

template <typename T, int dim>
int test_copy_offset(const sycl::range<dim> &DstOffset,
                     const sycl::range<dim> &SrcOffset) {
  const size_t rows = 5, cols = 4, Size = rows * cols;
  std::vector<T> Data(Size);
  std::iota(Data.begin(), Data.end(), 0);
  std::vector<T> Values(Size, T{});

  const sycl::range<dim> Window(2, 2);
  {
    range<dim> range_2D(rows, cols);
    buffer<T, dim> BufferFrom(&Data[0], range_2D);
    buffer<T, dim> BufferTo(&Values[0], range_2D);
    queue Queue;

    Queue.submit([&](handler &Cgh) {
      accessor<T, dim, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, Window, SrcOffset);
      accessor<T, dim, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, Window, DstOffset);
      Cgh.copy(AccessorFrom, AccessorTo);
    });
  }
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      const T &dval = Values[r * cols + c];
      if (r >= DstOffset[0] && r < DstOffset[0] + Window[0] &&
          c >= DstOffset[1] && c < DstOffset[1] + Window[1]) {

        const size_t r1 = r - DstOffset[0] + SrcOffset[0];
        const size_t c1 = c - DstOffset[1] + SrcOffset[1];
        const T &sval = Data[r1 * cols + c1];
        if (dval != sval)
          return 1;
      } else if (dval != T())
        return 1;
    }
  }
  return 0;
}

template <typename T> int test_copy_offset() {
  for (int SO1 = 0; SO1 < 3; SO1++)
    for (int SO2 = 0; SO2 < 3; SO2++)
      for (int DO1 = 0; DO1 < 3; DO1++)
        for (int DO2 = 0; DO2 < 3; DO2++) {
          const sycl::range<2> SrcOffset(SO1, SO2);
          const sycl::range<2> DstOffset(DO1, DO2);
          if (int ret = test_copy_offset<T>(DstOffset, SrcOffset))
            return ret;
        }
  return 0;
}

// todo: different dimensions
int main() {
  if (int r = test_copy_offset<int>())
    return r;
  return 0;
}
