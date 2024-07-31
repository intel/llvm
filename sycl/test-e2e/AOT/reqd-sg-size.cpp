// This test ensures that a program that has a kernel
// using various required sub-group sizes can be compiled AOT.

// REQUIRES: ocloc, opencl-aot, any-device-is-cpu
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_tgllp -o %t.tgllp.out %s
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 -o %t.x86.out %s

// ocloc on windows does not have support for PVC/CFL, so this command will
// result in an error when on windows. (In general, there is no support
// for pvc/cfl on windows.)
// RUN: %if !windows %{ %clangxx -fsycl -fsycl-targets=intel_gpu_cfl -o %t.cfl.out %s %}
// RUN: %if !windows %{ %clangxx -fsycl -fsycl-targets=intel_gpu_pvc -o %t.pvc.out %s %}

#include <cstdio>
#include <iostream>

#include <sycl/detail/core.hpp>

using namespace sycl;

template <int N> class kernel_name;

template <size_t... Ns> struct SubgroupDispatcher {
  std::vector<std::pair<size_t, size_t>> fails;
  SubgroupDispatcher(queue &q) : q(q) {}

  void operator()(const std::vector<size_t> &v) {
    for (auto i : v)
      (*this)(i);
  }

  void operator()(size_t n) { (dispatch<Ns>(n), ...); }

private:
  queue &q;

  template <size_t size> void dispatch(size_t n) {
    if (n == size) {
      size_t res = 0;
      {
        buffer<size_t, 1> buf(&res, 1);
        q.submit([&](handler &cgh) {
          accessor acc{buf, cgh};
          cgh.parallel_for<kernel_name<size>>(
              nd_range<1>(1, 1),
              [=](auto item) [[intel::reqd_sub_group_size(size)]] {
                acc[0] = item.get_sub_group().get_max_local_range()[0];
              });
        });
      }
      if (res != size)
        fails.push_back({res, size});
    }
  }
};

int main() {
  queue q;
  auto ctx = q.get_context();
  auto dev = q.get_device();
  auto sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
  std::cout << "  sub-group sizes supported by the device: " << sizes[0];
  for (int i = 1; i < sizes.size(); ++i) {
    std::cout << ", " << sizes[i];
  }
  std::cout << '\n';

  using dispatcher_t = SubgroupDispatcher<4, 8, 16, 32, 64, 128>;
  dispatcher_t dispatcher(q);
  dispatcher(sizes);
  if (dispatcher.fails.size() > 0) {
    for (auto [actual, expected] : dispatcher.fails) {
      std::cout << "actual:   " << actual << "\n"
                << "expected: " << expected << "\n";
    }
  } else {
    std::cout << "pass\n";
  }
}
