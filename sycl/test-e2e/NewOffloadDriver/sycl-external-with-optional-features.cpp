// REQUIRES: pdtracker
// PDTRACKER: https://github.com/intel/llvm/issues/18432
// Test with `--offload-new-driver`
// RUN: %{build} -DSOURCE1 --offload-new-driver -c -o %t1.o
// RUN: %{build} -DSOURCE2 --offload-new-driver -c -o %t2.o
// RUN: %clangxx -Wno-error=unused-command-line-argument -fsycl %{sycl_target_opts} --offload-new-driver %t1.o %t2.o -o %t.exe
// RUN: %{run} %t.exe

#ifdef SOURCE1
#include <iostream>
#include <sycl/detail/core.hpp>

using accT = sycl::accessor<int, 1>;
constexpr int value = 42;

template <sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL void func(const accT &acc);

int main() {
  sycl::queue q;
  int data = 0;
  sycl::buffer<int> buf{&data, {1}};
  if (q.get_device().has(sycl::aspect::cpu)) {
    q.submit([&](sycl::handler &cgh) {
       accT acc{buf, cgh};
       cgh.single_task<class Foo>([=] { func<sycl::aspect::cpu>(acc); });
     }).wait_and_throw();
  } else if (q.get_device().has(sycl::aspect::gpu)) {
    q.submit([&](sycl::handler &cgh) {
       accT acc{buf, cgh};
       cgh.single_task<class Bar>([=] { func<sycl::aspect::gpu>(acc); });
     }).wait_and_throw();
  }
  std::cout << "OK" << std::endl;
}

#endif // SOURCE1

#ifdef SOURCE2
#include <sycl/detail/core.hpp>

constexpr int value = 42;

using accT = sycl::accessor<int, 1>;

template <sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL void func(const accT &acc);
template <> SYCL_EXTERNAL void func<sycl::aspect::cpu>(const accT &acc) {
  acc[0] = value;
}
template <> SYCL_EXTERNAL void func<sycl::aspect::gpu>(const accT &acc) {
  acc[0] = value;
}

#endif // SOURCE2
