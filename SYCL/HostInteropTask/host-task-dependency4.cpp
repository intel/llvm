// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

cl::sycl::event submit(cl::sycl::queue &Q, cl::sycl::buffer<int> &B) {
  return Q.submit([&](cl::sycl::handler &CGH) {
    auto A = B.template get_access<cl::sycl::access::mode::read_write>(CGH);
    CGH.host_task([=]() { (void)A; });
  });
}

int main() {
  cl::sycl::queue Q;
  int Status = 0;
  cl::sycl::buffer<int> A{&Status, 1};
  std::vector<cl::sycl::event> Events;

  Events.push_back(submit(Q, A));
  Events.push_back(submit(Q, A));
  Q.submit([&](sycl::handler &CGH) {
     CGH.depends_on(Events);
     CGH.host_task([&] { printf("all done\n"); });
   }).wait_and_throw();

  return 0;
}
