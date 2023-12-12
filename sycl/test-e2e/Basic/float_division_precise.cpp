// RUN: %{build} -ffp-model=precise -o %t.out
// RUN: %{run} %t.out

// Tests that -ffp-model=precise causes floating point division to be the same
// on device and host.

#include <sycl.hpp>

constexpr size_t NumElems = 1024;

int main() {
  sycl::queue Q;
  float *InData = sycl::malloc_shared<float>(NumElems, Q);
  float *OutData = sycl::malloc_shared<float>(NumElems, Q);

  for (size_t I = 0; I < NumElems; ++I) {
    InData[I] = float(I) + 1.0f;
    OutData[I] = 0.0f;
  }

  Q.parallel_for(sycl::range<1>(NumElems), [=](sycl::id<1> Idx) {
     OutData[Idx] = InData[Idx] / InData[NumElems - Idx - 1];
   }).wait_and_throw();

  size_t NumFails = 0;
  for (size_t I = 0; I < NumElems; ++I) {
    float Expected = InData[I] / InData[NumElems - I - 1];
    if (OutData[I] != Expected) {
      std::cout << "Unexpected result for element " << I << ": " << OutData[I]
                << " != " << Expected << std::endl;
      ++NumFails;
    }
  }

  sycl::free(InData, Q);
  sycl::free(OutData, Q);

  return NumFails;
}
