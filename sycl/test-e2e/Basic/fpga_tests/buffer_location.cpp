// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Queue;
  sycl::buffer<int, 1> Buf{sycl::range{1}};

  Queue.submit([&](sycl::handler &CGH) {
    sycl::ext::oneapi::accessor_property_list PL{
        sycl::ext::intel::buffer_location<1>};
    sycl::accessor Acc(Buf, CGH, sycl::write_only, PL);
    CGH.single_task<class Test>([=]() { Acc[0] = 42; });
  });

  Queue.wait();

  auto Acc = Buf.template get_access<sycl::access::mode::read_write>();
  assert(Acc[0] == 42 && "Value mismatch");

  return 0;
}
