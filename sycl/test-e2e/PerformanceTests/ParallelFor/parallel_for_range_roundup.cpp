// RUN: echo "Running parallel_for benchmark without range rounding"
// RUN: %{build} -fsycl-range-rounding=disable -o %t.out
// RUN: %{run} %t.out

// RUN: echo "Running parallel_for benchmark with normal range rounding"
// RUN: %{build} -fsycl-range-rounding=force -o %t.out
// RUN: %{run} %t.out

// RUN: echo "Running parallel_for benchmark with experimental range rounding"
// RUN: %{build} -fsycl-exp-range-rounding -fsycl-range-rounding=force -o %t.out
// RUN: %{run} %t.out

#include <chrono>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

class FillData;
class Compute;

int main() {
  constexpr static size_t width{788};
  constexpr static size_t height{1888};
  constexpr static size_t N{width * height};
  constexpr static size_t iterations{1000};

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
         });
   }).wait_and_throw();

  auto start{std::chrono::steady_clock::now()};
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
  auto end{std::chrono::steady_clock::now()};
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;
}
