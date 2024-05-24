// The test checks that ESIMD kernels support SYCL 2020 specialization constants
// for all basic types, particularly a specialization constant can be redifined
// and correct new value is used after redefinition.

#include "esimd_test_utils.hpp"

#include <sycl/specialization_id.hpp>

using namespace sycl;

template <typename AccessorTy>
ESIMD_INLINE void do_store(AccessorTy acc, int i, spec_const_t val) {
  using namespace sycl::ext::intel::esimd;
  // scatter function, that is used in scalar_store, can only process types
  // whose size is no more than 4 bytes.
#if (STORE == 0)
  // bool
  scalar_store<container_t>(acc, i * sizeof(container_t), val ? 1 : 0);
#elif (STORE == 1)
  // block
  block_store<container_t>(acc, i * sizeof(container_t),
                           simd<spec_const_t, 2>{val});
#else
  static_assert(STORE == 2, "Unspecified store");
  // scalar
  scalar_store<container_t>(acc, i * sizeof(container_t), val);
#endif
}

class TestKernel;

constexpr specialization_id<spec_const_t> ConstID(DEF_VAL);

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  std::vector<container_t> etalon = {DEF_VAL, REDEF_VAL};
  const size_t n_times = etalon.size();
  std::vector<container_t> output(n_times);

  bool passed = true;
  for (int i = 0; i < n_times; i++) {
    try {
      sycl::buffer<container_t, 1> buf(output.data(), output.size());

      q.submit([&](sycl::handler &cgh) {
         auto acc = buf.get_access<sycl::access::mode::write>(cgh);
         if (i % 2 != 0)
           cgh.set_specialization_constant<ConstID>(REDEF_VAL);
         cgh.single_task<TestKernel>([=](kernel_handler kh) SYCL_ESIMD_KERNEL {
           do_store(acc, i, kh.get_specialization_constant<ConstID>());
         });
       }).wait();
    } catch (sycl::exception const &e) {
      std::cout << "SYCL exception caught: " << e.what() << '\n';
      return 1;
    }

    if (output[i] != etalon[i]) {
      passed = false;
      std::cout << "comparison error -- case #" << i << " -- ";
      std::cout << "output: " << output[i] << ", ";
      std::cout << "etalon: " << etalon[i] << std::endl;
    }
  }

  if (passed) {
    std::cout << "passed" << std::endl;
    return 0;
  }

  std::cout << "FAILED" << std::endl;
  return 1;
}
