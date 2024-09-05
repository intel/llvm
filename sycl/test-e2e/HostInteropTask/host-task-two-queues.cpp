// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// TODO: Flaky fail on Level Zero that is why mark as unsupported temporarily.
// UNSUPPORTED: level_zero

#include <iostream>
#include <sycl/detail/core.hpp>
#include <vector>

namespace S = sycl;

#define WIDTH 5
#define HEIGHT 5

void test() {
  auto EH = [](S::exception_list EL) {
    for (const std::exception_ptr &E : EL) {
      throw E;
    }
  };

  S::queue Q1(EH);
  S::queue Q2(EH);

  std::vector<int> DataA(WIDTH * HEIGHT, 2);
  std::vector<int> DataB(WIDTH * HEIGHT, 3);
  std::vector<int> DataC(WIDTH * HEIGHT, 1);

  S::buffer<int, 2> BufA{DataA.data(), S::range<2>{WIDTH, HEIGHT}};
  S::buffer<int, 2> BufB{DataB.data(), S::range<2>{WIDTH, HEIGHT}};
  S::buffer<int, 2> BufC{DataC.data(), S::range<2>{WIDTH, HEIGHT}};

  auto CG1 = [&](S::handler &CGH) {
    auto AccA = BufA.get_access<S::access::mode::read>(CGH);
    auto AccB = BufB.get_access<S::access::mode::read>(CGH);
    auto AccC = BufC.get_access<S::access::mode::read_write>(CGH);
    auto Kernel = [=](S::nd_item<2> Item) {
      size_t W = Item.get_global_id(0);
      size_t H = Item.get_global_id(1);
      AccC[W][H] += AccA[W][H] * AccB[W][H];
    };
    CGH.parallel_for<class K1>(S::nd_range<2>({WIDTH, HEIGHT}, {1, 1}), Kernel);
  };

  auto CG2 = [&](S::handler &CGH) {
    auto AccA = BufA.get_access<sycl::access::mode::read>(CGH);
    auto AccB = BufB.get_access<sycl::access::mode::read>(CGH);
    auto AccC = BufC.get_access<sycl::access::mode::read_write>(CGH);

    CGH.host_task([=] {
      for (size_t I = 0; I < WIDTH; ++I)
        for (size_t J = 0; J < HEIGHT; ++J) {
          std::cout << "C[" << I << "][" << J << "] = " << AccC[I][J]
                    << std::endl;
        }
    });
  };

  static const size_t NTIMES = 4;

  for (size_t Idx = 0; Idx < NTIMES; ++Idx) {
    Q1.submit(CG1);
    Q2.submit(CG2);
    Q2.submit(CG1);
    Q1.submit(CG2);
  }

  Q1.wait_and_throw();
  Q2.wait_and_throw();

  for (size_t I = 0; I < WIDTH; ++I)
    for (size_t J = 0; J < HEIGHT; ++J)
      assert(DataC[I * HEIGHT + J] == (1 + 2 * 3 * NTIMES * 2));
}

int main(void) {
  test();
  return 0;
}
