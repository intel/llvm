// UNSUPPORTED: aspect-fp64
// RUN: %{build} -o %t.out -O3
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

using namespace sycl;

template <aspect asp, typename T>
[[sycl::device_has(asp)]] void dummy_function_decorated(const T &acc) {
  acc[0] = true;
}

int main() {
  queue q;
  bool b = false;
  assert(!q.get_device().has(aspect::fp64));

  buffer<bool, 1> buf(&b, 1);
  try {
    q.submit([&](handler &cgh) {
      accessor acc(buf, cgh);
      cgh.single_task([=]() { dummy_function_decorated<aspect::fp64>(acc); });
    });
    std::cout << "Exception should have been thrown!\n";
    return 1;
  } catch (const sycl::exception &e) {
    if (e.code() != errc::kernel_not_supported) {
      std::cout << "Exception caught, but wrong error code!\n";
      throw;
    }
    std::cout << "pass\n";
    return 0;
  }
}
