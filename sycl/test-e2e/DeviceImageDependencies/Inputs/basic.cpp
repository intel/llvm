#include "a.hpp"

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <cassert>
#include <functional>
#include <iostream>

using namespace sycl;

template <int N> class Kernel;
template <typename T> void runTest(queue &q, T SubmitOp) {
  int val = 0;
  {
    buffer<int, 1> buf(&val, range<1>(1));
    SubmitOp(q, buf);
  }
  std::cout << "val=" << std::hex << val << "\n";
  assert(val == 0xDCBA);
}

int main() {
  queue q;
  runTest(q, [](queue &q, buffer<int, 1> &buf) {
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access(cgh);
      cgh.single_task<Kernel<1>>([=]() { acc[0] = levelA(acc[0]); });
    });
  });
  runTest(q, [](queue &q, buffer<int, 1> &buf) {
    kernel_bundle KB = get_kernel_bundle<sycl::bundle_state::executable>(
        q.get_context(), {get_kernel_id<Kernel<2>>()});
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access(cgh);
      cgh.use_kernel_bundle(KB);
      cgh.single_task<Kernel<2>>([=]() { acc[0] = levelA(acc[0]); });
    });
  });
  runTest(q, [](queue &q, buffer<int, 1> &buf) {
    kernel_bundle KBInput = get_kernel_bundle<sycl::bundle_state::input>(
        q.get_context(), {get_kernel_id<Kernel<3>>()});
    kernel_bundle KBObject = compile(KBInput);
    kernel_bundle KBLinked = link(KBObject);
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access(cgh);
      cgh.use_kernel_bundle(KBLinked);
      cgh.single_task<Kernel<3>>([=]() { acc[0] = levelA(acc[0]); });
    });
  });
}
