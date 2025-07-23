#include <sycl/sycl.hpp>

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size) {
  for (size_t I = 0; I < Size; ++I)
    Ptr[I] = I;
}
