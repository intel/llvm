// REQUIRES: opencl-aot, cpu
//
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
//
// The test checks that the specialization constant feature works with ahead
// of time compilation.

#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>

class MyInt32Const;

using namespace sycl;

class Kernel;

int main(int argc, char **argv) {
  sycl::queue q(default_selector{}, [](exception_list l) {
    for (auto ep : l) {
      try {
        std::rethrow_exception(ep);
      } catch (sycl::exception &e0) {
        std::cout << e0.what();
      } catch (std::exception &e1) {
        std::cout << e1.what();
      } catch (...) {
        std::cout << "*** catch (...)\n";
      }
    }
  });

  std::cout << "Running on " << q.get_device().get_info<info::device::name>()
            << "\n";
  sycl::program prog(q.get_context());

  sycl::ext::oneapi::experimental::spec_constant<int32_t, MyInt32Const> i32 =
      prog.set_spec_constant<MyInt32Const>(10);

  prog.build_with_kernel_type<Kernel>();

  std::vector<int> vec(1);
  {
    sycl::buffer<int, 1> buf(vec.data(), vec.size());

    q.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task<Kernel>(prog.get_kernel<Kernel>(),
                              [=]() { acc[0] = i32.get(); });
    });
  }
  bool passed = true;
  int val = vec[0];
  int gold = 0; // with AOT, spec constant is set to C++ default for the type

  if (val != gold) {
    std::cout << "*** ERROR: " << val << " != " << gold << "(gold)\n";
    passed = false;
  }
  std::cout << (passed ? "passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
