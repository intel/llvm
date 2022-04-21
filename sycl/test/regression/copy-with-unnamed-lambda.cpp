// RUN: %clangxx -fsycl -fsycl-unnamed-lambda %s -o %t.out
// The purpose of this test is to check that the following code can be
// successfully compiled
#include <CL/sycl.hpp>

#include <iostream>

int main() {
  auto AsyncHandler = [](cl::sycl::exception_list EL) {
    for (std::exception_ptr const &P : EL) {
      try {
        std::rethrow_exception(P);
      } catch (std::exception const &E) {
        std::cerr << "Caught async SYCL exception: " << E.what() << std::endl;
      }
    }
  };

  cl::sycl::queue Q(AsyncHandler);

  constexpr size_t Size = 10;
  const int ReferenceData[Size] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

  cl::sycl::buffer<int> Buf(Size);

  Q.submit([&](cl::sycl::handler &CGH) {
    auto Acc = Buf.get_access<cl::sycl::access::mode::write>(CGH);
    CGH.copy(ReferenceData, Acc);
  });

  Q.wait_and_throw();

  auto Acc = Buf.get_access<cl::sycl::access::mode::read>();
  for (size_t I = 0; I < Size; ++I) {
    if (ReferenceData[I] != Acc[I]) {
      std::cerr << "Incorrect result, got: " << Acc[I]
                << ", expected: " << ReferenceData[I] << std::endl;
      return 1;
    }
  }

  int CopybackData[Size] = { 0 };
  Q.submit([&](cl::sycl::handler &CGH) {
    auto Acc = Buf.get_access<cl::sycl::access::mode::read>(CGH);
    CGH.copy(Acc, CopybackData);
  });

  Q.wait_and_throw();

  for (size_t I = 0; I < Size; ++I) {
    if (ReferenceData[I] != CopybackData[I]) {
      std::cerr << "Incorrect result, got: " << Acc[I]
                << ", expected: " << ReferenceData[I] << std::endl;
      return 1;
    }
  }

  return 0;
}
