// RUN: %clangxx -fsycl -fsyntax-only -sycl-std=2020 %s

// Tests the return type of combiner operations on reducers.

#include <sycl/sycl.hpp>

#include <type_traits>

int main() {
  sycl::queue Q;

  int *PlusMem = sycl::malloc_device<int>(1, Q);
  int *MultMem = sycl::malloc_device<int>(1, Q);
  int *BitAndMem = sycl::malloc_device<int>(1, Q);
  int *BitOrMem = sycl::malloc_device<int>(1, Q);
  int *BitXorMem = sycl::malloc_device<int>(1, Q);
  Q.submit([&](sycl::handler &CGH) {
    auto PlusReduction = sycl::reduction(PlusMem, sycl::plus<int>());
    auto MultReduction = sycl::reduction(MultMem, sycl::multiplies<int>());
    auto BitAndReduction = sycl::reduction(BitAndMem, sycl::bit_and<int>());
    auto BitOrReduction = sycl::reduction(BitOrMem, sycl::bit_or<int>());
    auto BitXorReduction = sycl::reduction(PlusMem, sycl::bit_xor<int>());
    CGH.parallel_for(sycl::range<1>(10), PlusReduction, MultReduction,
                     BitAndReduction, BitOrReduction, BitXorReduction,
                     [=](sycl::id<1>, auto &Plus, auto &Mult, auto &BitAnd,
                         auto &BitOr, auto &BitXor) {
                       (Plus.combine(1) += 1).combine(1);
                       (Plus.combine(1)++).combine(1);
                       (++Plus.combine(1)).combine(1);
                       (Mult.combine(1) *= 1).combine(1);
                       (BitAnd.combine(1) &= 1).combine(1);
                       (BitOr.combine(1) |= 1).combine(1);
                       (BitXor.combine(1) ^= 1).combine(1);
                     });
  });
  sycl::free(PlusMem, Q);
  sycl::free(MultMem, Q);
  sycl::free(BitAndMem, Q);
  sycl::free(BitOrMem, Q);
  sycl::free(BitXorMem, Q);
  return 0;
}
