// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note -o - %s
#include <iostream>
#include <sycl/sycl.hpp>
int main() {
  {
    int varA = 42;
    int varB = 42;
    int sum = 0;
    sycl::queue myQueue{};
    {
      myQueue
          .single_task([&](sycl::handler &cgh) {
            // expected-error-re@sycl/queue.hpp:* {{static assertion failed due to requirement '{{.*}}': sycl::queue.single_task() requires a kernel instead of command group.{{.*}} Use queue.submit() instead}}
            // expected-error-re@sycl/detail/cg_types.hpp:* {{no matching function for call to object of type '(lambda at {{.*}}single_task_error_message.cpp:{{.*}})'}}
          })
          .wait();
    }
  }
  {
    int varA = 42;
    int varB = 42;
    int sum = 0;
    sycl::queue myQueue{};
    {
      sycl::event e{};
      myQueue
          .single_task(e,
                       [&](sycl::handler &cgh) {
                         // expected-error-re@sycl/queue.hpp:* {{static assertion failed due to requirement '{{.*}}': sycl::queue.single_task() requires a kernel instead of command group.{{.*}} Use queue.submit() instead}}
                         // expected-error-re@sycl/detail/cg_types.hpp:* {{no matching function for call to object of type '(lambda at {{.*}}single_task_error_message.cpp:{{.*}})'}}
                       })
          .wait();
    }
  }
  {
    int varA = 42;
    int varB = 42;
    int sum = 0;
    sycl::queue myQueue{};
    {
      std::vector<sycl::event> vector_event;
      myQueue
          .single_task(vector_event,
                       [&](sycl::handler &cgh) {
                         // expected-error-re@sycl/queue.hpp:* {{static assertion failed due to requirement '{{.*}}': sycl::queue.single_task() requires a kernel instead of command group.{{.*}} Use queue.submit() instead}}
                         // expected-error-re@sycl/detail/cg_types.hpp:* {{no matching function for call to object of type '(lambda at {{.*}}single_task_error_message.cpp:{{.*}})'}}
                       })
          .wait();
    }
  }
}
