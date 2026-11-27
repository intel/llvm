// REQUIRES: native_cpu
// RUN: %{build} -Xclang -fsycl-allow-func-ptr -O2 -o %t.out
// RUN: %{run} %t.out

// Check that an indirect call whose callee differs between work-items calls
// the right function for each work-item once the kernel has been vectorized.
// The matching device-code check lives in
// sycl/test/check_device_code/native_cpu/vecz-indirect-call.cpp.

#include <iostream>
#include <sycl/sycl.hpp>

using fn_t = int (*)(int);
SYCL_EXTERNAL int add1(int x) { return x + 1; }
SYCL_EXTERNAL int mul2(int x) { return x * 2; }

int main() {
  sycl::queue q;
  const int wanted[16] = {0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0};
  sycl::buffer<int, 1> sel(sycl::range<1>(16));
  {
    sycl::host_accessor s(sel, sycl::write_only);
    for (int i = 0; i < 16; ++i)
      s[i] = wanted[i];
  }
  sycl::buffer<int, 1> out(sycl::range<1>(16));
  q.submit([&](sycl::handler &h) {
    sycl::accessor s(sel, h, sycl::read_only);
    sycl::accessor o(out, h, sycl::write_only);
    h.parallel_for<class IndirectCall>(
        sycl::nd_range<1>(sycl::range<1>(16), sycl::range<1>(8)),
        [=](sycl::nd_item<1> item) {
          sycl::id<1> i = item.get_global_id();
          fn_t f = s[i] ? add1 : mul2;
          o[i] = f(i[0] + 3); // + 3 because 1 + 1 == 1 * 2
        });
  });
  sycl::host_accessor o(out, sycl::read_only);
  int error = 0;
  for (int i = 0; i < 16; ++i) {
    const auto expected = (wanted[i] ? i + 4 : (i + 3) * 2);
    if (o[i] != expected) {
      std::cerr << "Mismatch at index " << i << ": got " << o[i]
                << ", expected " << expected << '\n';
      ++error;
    }
  }
  return error;
}
