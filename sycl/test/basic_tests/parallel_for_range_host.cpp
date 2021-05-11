// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

int main() {
  auto AsyncHandler = [](exception_list ES) {
    for (auto& E : ES) {
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

  // parallel_for, 100 global, 3 local -> fail.
  try {
    Q.submit([&](handler &CGH) {
        CGH.parallel_for<class a>(nd_range<1>(range<1>(100), range<1>(3)),
            [=](nd_item<1>) {});
    });
    Q.wait_and_throw();
    std::cerr << "Test case 'a' failed: no exception has been thrown"
              << std::endl;
    return 1;
  } catch (nd_range_error) {
    // We expect an error to be thrown!
  }

  // parallel_for, 100 global, 4 local -> pass.
  try {
    Q.submit([&](handler &CGH) {
        CGH.parallel_for<class b>(nd_range<1>(range<1>(100), range<1>(4)),
            [=](nd_item<1>) {});
    });
    Q.wait_and_throw();
  } catch (nd_range_error) {
    std::cerr << "Test case 'b' failed: exception has been thrown" << std::endl;
    return 1;
  }

  // parallel_for, (100, 33, 16) global, (2, 3, 4) local -> pass.
  try {
    Q.submit([&](handler &CGH) {
        CGH.parallel_for<class c>(nd_range<3>(range<3>(100, 33, 16),
              range<3>(2, 3, 4)),
            [=](nd_item<3>) {});
    });
    Q.wait_and_throw();
  } catch (nd_range_error) {
    std::cerr << "Test case 'c' failed: exception has been thrown" << std::endl;
    return 1;
  }

  // parallel_for, (100, 33, 16) global, (2, 3, 5) local -> fail.
  try {
    Q.submit([&](handler &CGH) {
        CGH.parallel_for<class d>(nd_range<3>(range<3>(100, 33, 16),
              range<3>(2, 3, 5)),
            [=](nd_item<3>) {});
    });
    Q.wait_and_throw();
    std::cerr << "Test case 'd' failed: no exception has been thrown"
              << std::endl;
    return 1;
  } catch (nd_range_error) {
  }

  // local size has a 0-based range -- no SIGFPEs, we hope.
  try {
    Q.submit([&](handler &CGH) {
        CGH.parallel_for<class e>(nd_range<2>(range<2>(5, 33), range<2>(1, 0)),
            [=](nd_item<2>) {});
    });
    Q.wait_and_throw();
    std::cerr << "Test case 'e' failed: no exception has been thrown"
              << std::endl;
    return 1;
  } catch (nd_range_error) {
  }

  // parallel_for_work_group with 0-based local range.
  try {
    Q.submit([&](handler &CGH) {
        CGH.parallel_for_work_group<class f>(range<2>(5, 33), range<2>(1, 0),
            [=](group<2>) {});
    });
    Q.wait_and_throw();
    std::cerr << "Test case 'f' failed: no exception has been thrown"
              << std::endl;
    return 1;
  } catch (nd_range_error) {
  }

  // parallel_for, 30 global, 1(implicit) local -> pass.
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class g>(range<1>(30),
                                [=](id<1>) {});
    });
    Q.wait_and_throw();
  } catch (nd_range_error) {
    std::cerr << "Test case 'g' failed: exception has been thrown" << std::endl;
    return 1;
  }
  return 0;
}
