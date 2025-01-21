// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

// This short test simply confirms that target::device
// is supported correctly.
// TODO: delete this test and test the functionality
// over in llvm-test-suite along with the other changes
// needed to support the SYCL 2020 target updates.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue testQueue;
  sycl::range<1> ndRng(1);
  int kernelResult;
  {
    sycl::buffer<int, 1> buffer(&kernelResult, ndRng);

    testQueue.submit([&](sycl::handler &cgh) {
      auto ptr = buffer.get_access<sycl::access_mode::read_write,
                                   sycl::target::device>(cgh);
      cgh.single_task<class kernel>([=]() { ptr[0] = 5; });
    });
  } // ~buffer

  // std::cout << "kernelResult should be 5: " << kernelResult << std::endl;
  assert(kernelResult == 5);

  return 0;
}