// XFAIL: gpu-intel-gen12
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/17872
// RUN: %{build} -fsycl-range-rounding=disable -o %t1.out
// RUN: %{run} %t1.out --variant="rounding=disable"

// RUN: %{build} -fsycl-range-rounding=force -o %t2.out
// RUN: %{run} %t2.out --variant="rounding=force"

// RUN: %{build} -fsycl-exp-range-rounding -fsycl-range-rounding=force -o %t3.out
// RUN: %{run} %t3.out --variant="rounding=exp+force"

// PERF-DESCRIPTION: 1000 submits of a 2D parallel_for per range rounding mode.

#include "../perf_harness.hpp"

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <algorithm>
#include <vector>

class FillData;
class Compute;

int main(int argc, char **argv) {
  constexpr static size_t width{788};
  constexpr static size_t height{1888};
  constexpr static size_t N{width * height};
  constexpr static size_t iterations{1000};

  perf::add_case("range<2> 1888x788, 1000 submits", "ms", [&](perf::State &S) {
    sycl::queue q{};
    float *A{sycl::malloc_device<float>(N, q)};
    float *B{sycl::malloc_device<float>(N, q)};
    float *C{sycl::malloc_device<float>(N, q)};

    q.submit([&](sycl::handler &cgh) {
       cgh.parallel_for<FillData>(
           sycl::range<2>{height, width}, [=](sycl::id<2> id) {
             unsigned int row{static_cast<unsigned int>(id[0])};
             unsigned int col{static_cast<unsigned int>(id[1])};
             unsigned int ix{row * static_cast<unsigned int>(width) + col};
             A[ix] = id[0];
             B[ix] = id[1];
             C[ix] = -1;
           });
     }).wait_and_throw();

    S.measure([&] {
      for (size_t i{0}; i < iterations; ++i) {
        q.submit([&](sycl::handler &cgh) {
           cgh.parallel_for<Compute>(
               sycl::range<2>{height, width}, [=](sycl::id<2> id) {
                 unsigned int row{static_cast<unsigned int>(id[0])};
                 unsigned int col{static_cast<unsigned int>(id[1])};
                 unsigned int ix{row * static_cast<unsigned int>(width) + col};
                 if (ix >= static_cast<unsigned int>(N)) {
                   return;
                 }
                 if (A[ix] > B[ix]) {
                   C[ix] = A[ix];
                 } else {
                   C[ix] = B[ix];
                 }
               });
         }).wait_and_throw();
      }
    });

    std::vector<float> Host(N);
    q.memcpy(Host.data(), C, N * sizeof(float)).wait_and_throw();
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);
    for (size_t row = 0; row < height; ++row)
      for (size_t col = 0; col < width; ++col)
        perf::check(Host[row * width + col] == std::max(row, col),
                    "C[ix] == max(row, col)");
  });

  return perf::main(argc, argv);
}
