// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsyntax-only -fsycl-targets=spir64_fpga %s

// When using zero dimension accessors with atomic access we
// want to make sure they are compiling correctly on all devices,
// especially FPGA which changes some of the template specializations
// with the __ENABLE_USM_ADDR_SPACE__ macro.

#include <sycl/sycl.hpp>

using namespace sycl;

using atomic_t = sycl::atomic<int>;

// store() is defined for both int and atomic
void store(int &foo, int value) { foo = value; }

void store(atomic_t foo, int value) { foo.store(value); }

int main(int argc, char *argv[]) {

  queue q(default_selector_v);

  // Accessor with dimensionality 0.
  {
    try {
      int data = -1;
      int atomic_data = -1;
      {
        sycl::buffer<int, 1> b(&data, sycl::range<1>(1));
        sycl::buffer<int, 1> atomic_b(&atomic_data, sycl::range<1>(1));
        sycl::queue queue;
        queue.submit([&](sycl::handler &cgh) {
          sycl::accessor<int, 0, sycl::access::mode::read_write,
                         sycl::access::target::global_buffer>
              NormalA(b, cgh);
          sycl::accessor<int, 0, sycl::access::mode::atomic,
                         sycl::access::target::global_buffer>
              AtomicA(atomic_b, cgh);
          cgh.single_task<class acc_with_zero_dim>([=]() {
            // 'normal int'
            store(NormalA, 399);

            // 'atomic int'
            store(AtomicA, 499);
            // This error is the one we do NOT want to see when compiling on
            // FPGA
            // clang-format off
                    // error: no matching function for call to 'store'
                    // note: candidate function not viable: no known conversion from 'const sycl::accessor<int, 0, sycl::access::mode::atomic, sycl::access::target::global_buffer>' to 'int &' for 1st argument
                    // note: candidate function not viable: no known conversion from 'const sycl::accessor<int, 0, sycl::access::mode::atomic, sycl::access::target::global_buffer>' to 'atomic_t' (aka 'atomic<int>') for 1st argument
            // clang-format on
          });
        });
      }
      assert(data == 399);
      assert(atomic_data == 499);
    } catch (sycl::exception e) {
      std::cout << "SYCL exception caught: " << e.what();
      return 1;
    }
  }
  std::cout << std::endl;

  return 0;
}
