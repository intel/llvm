// UNSUPPORTED: cuda || hip
//
// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks that the specialization constant feature works correctly with
// composite types: toolchain processes them correctly and runtime can correctly
// execute the program.

#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>

using namespace sycl;
class Test;

struct A {
  int a;
  float b;
};

struct POD {
  A a[2];
  int b;
};

using MyPODConst = POD;

int global_val = 10;

// Fetch a value at runtime.
int get_value() { return global_val; }

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
  std::cout << "global_val = " << global_val << "\n";
  sycl::program program(q.get_context());

  int goldi = (int)get_value();
  float goldf = (float)get_value();

  POD gold = {{{goldi, goldf}, {goldi, goldf}}, goldi};

  sycl::ext::oneapi::experimental::spec_constant<POD, MyPODConst> pod =
      program.set_spec_constant<MyPODConst>(gold);

  program.build_with_kernel_type<Test>();

  POD result;
  try {
    sycl::buffer<POD, 1> bufi(&result, 1);

    q.submit([&](sycl::handler &cgh) {
      auto acci = bufi.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task<Test>(program.get_kernel<Test>(),
                            [=]() { acci[0] = pod.get(); });
    });
  } catch (sycl::exception &e) {
    std::cout << "*** Exception caught: " << e.what() << "\n";
    return 1;
  }

  bool passed = false;

  std::cout << result.a[0].a << " " << result.a[0].b << "\n";
  std::cout << result.a[1].a << " " << result.a[1].b << "\n";
  std::cout << result.b << "\n\n";

  std::cout << gold.a[0].a << " " << gold.a[0].b << "\n";
  std::cout << gold.a[1].a << " " << gold.a[1].b << "\n";
  std::cout << gold.b << "\n\n";

  if (0 == std::memcmp(&result, &gold, sizeof(POD))) {
    passed = true;
  }

  std::cout << (passed ? "passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
