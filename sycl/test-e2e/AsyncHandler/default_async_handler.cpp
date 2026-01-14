// RUN: %{build} -o %t.out
// RUN: %{run} %t.out 0 &> %t_0.txt ; FileCheck %s --input-file %t_0.txt --check-prefix CHECK-0
// RUN: %{run} %t.out 1 &> %t_1.txt ; FileCheck %s --input-file %t_1.txt --check-prefix CHECK-1
// RUN: %{run} %t.out 2 &> %t_2.txt ; FileCheck %s --input-file %t_2.txt --check-prefix CHECK-2

#include <string>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main(int argc, char *argv[]) {
  assert(argc == 2);
  int TestRun = std::stoi(argv[1]);
  assert(TestRun >= 0 && TestRun <= 3);

  queue Q;
  if (TestRun == 0) {
    Q.submit([&](handler &CGH) {
       CGH.host_task([=]() {
         throw std::runtime_error("Exception thrown from host_task through "
                                  "event::wait_and_throw().");
       });
     }).wait_and_throw();
  } else if (TestRun == 1) {
    Q.submit([&](handler &CGH) {
      CGH.host_task([=]() {
        throw std::runtime_error(
            "Exception thrown from host_task through queue::wait_and_throw().");
      });
    });
    Q.wait_and_throw();
  } else if (TestRun == 2) {
    Q.submit([&](handler &CGH) {
      CGH.host_task([=]() {
        throw std::runtime_error(
            "Exception thrown from host_task through queue::wait() and "
            "queue::throw_asynchronous().");
      });
    });
    Q.wait();
    Q.throw_asynchronous();
  }
  return 0;
}

// CHECK-0:      Default async_handler caught exceptions:
// CHECK-0-NEXT: Exception thrown from host_task through event::wait_and_throw().

// CHECK-1:      Default async_handler caught exceptions:
// CHECK-1-NEXT: Exception thrown from host_task through queue::wait_and_throw().

// CHECK-2:      Default async_handler caught exceptions:
// CHECK-2-NEXT: Exception thrown from host_task through queue::wait() and queue::throw_asynchronous().

// CHECK-3:      Default async_handler caught exceptions:
// CHECK-3-NEXT: Exception thrown from host_task through event::wait_and_throw() after queue death.
// CHECK-3-NOT:  Custom queue async handler was called!
