#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

using namespace sycl;
using namespace sycl::access;

void kernelFunc(int *Buf, int wiID) {
  Buf[wiID] = 0;
  assert(Buf[wiID] != 0 && "from assert statement");
}

void assertTest() {
  std::array<int, 4> Vec = {1, 2, 3, 4};
  sycl::range<1> numOfItems{Vec.size()};
  sycl::buffer<int, 1> Buf(Vec.data(), numOfItems);

  queue Q;
  Q.submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<mode::read_write>(CGH);

    CGH.parallel_for<class TheKernel>(
        numOfItems, [=](item<1> Item) { kernelFunc(&Acc[0], Item[0]); });
  });
  Q.wait();
}

int main(int Argc, const char *Argv[]) {

  assertTest();

  std::cout << "The test ended." << std::endl;
  return 0;
}
