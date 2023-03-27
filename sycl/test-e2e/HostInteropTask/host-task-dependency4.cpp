// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

sycl::event submit(sycl::queue &Q, sycl::buffer<int> &B) {
  return Q.submit([&](sycl::handler &CGH) {
    auto A = B.template get_access<sycl::access::mode::read_write>(CGH);
    CGH.host_task([=]() { (void)A; });
  });
}

int main() {
  sycl::queue Q;
  int Status = 0;
  sycl::buffer<int> A{&Status, 1};
  std::vector<sycl::event> Events;

  Events.push_back(submit(Q, A));
  Events.push_back(submit(Q, A));
  Q.submit([&](sycl::handler &CGH) {
     CGH.depends_on(Events);
     CGH.host_task([&] { printf("all done\n"); });
   }).wait_and_throw();

  return 0;
}
