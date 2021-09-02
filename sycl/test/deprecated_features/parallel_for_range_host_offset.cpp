// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

int main() {
  auto AsyncHandler = [](exception_list ES) {
    for (auto &E : ES) {
      std::rethrow_exception(E);
    }
  };

  queue Q(AsyncHandler);

  // parallel_for, offset
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class offset>(range<1>(1), id<1>(16),
                                     [=](id<1> ID) { assert(ID == 16); });
    });
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class offset_2D>(range<2>(1, 1), id<2>(16, 17),
                                        [=](id<2> ID) {
                                          assert(ID[0] == 16);
                                          assert(ID[1] == 17);
                                        });
    });
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class offset_3D>(range<3>(1, 1, 1), id<3>(16, 17, 18),
                                        [=](id<3> ID) {
                                          assert(ID[0] == 16);
                                          assert(ID[1] == 17);
                                          assert(ID[2] == 18);
                                        });
    });
    Q.wait_and_throw();
  } catch (nd_range_error) {
    std::cerr << "Test case 'offset' failed: exception has been thrown"
              << std::endl;
    return 1;
  }

  return 0;
}
