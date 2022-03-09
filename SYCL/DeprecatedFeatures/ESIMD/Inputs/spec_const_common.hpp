//==--------------- spec_const_common.h  - DPC++ ESIMD on-device test -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The test checks that ESIMD kernels support specialization constants for all
// basic types, particularly a specialization constant can be redifined and
// correct new value is used after redefinition.

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>
#include <vector>

using namespace cl::sycl;

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
  simd<spec_const_t, 2> vals{val};
  vals.copy_to(acc, i * sizeof(container_t));
#else
  static_assert(STORE == 2, "Unspecified store");
  // scalar
  scalar_store<container_t>(acc, i * sizeof(container_t), val);
#endif
}

class ConstID;
class TestKernel;

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  std::vector<container_t> etalon = {DEF_VAL, REDEF_VAL};
  const size_t n_times = etalon.size();
  std::vector<container_t> output(n_times);

  bool passed = true;
  for (int i = 0; i < n_times; i++) {
    try {
      sycl::program prg(q.get_context());

      // Checking that already initialized constant can be overwritten.
      // According to standards proposals:
      //   A cl::sycl::experimental::spec_constant object is considered
      //   initialized once the result of a cl::sycl::program::set_spec_constant
      //   is assigned to it.
      //   A specialization constant value can be overwritten if the program was
      //   not built before by recalling set_spec_constant with the same ID and
      //   the new value. Although the type T of the specialization constant
      //   must remain the same.
      auto spec_const = prg.set_spec_constant<ConstID>((spec_const_t)DEF_VAL);
      if (i % 2 != 0)
        spec_const = prg.set_spec_constant<ConstID>((spec_const_t)REDEF_VAL);

      prg.build_with_kernel_type<TestKernel>();

      sycl::buffer<container_t, 1> buf(output.data(), output.size());

      q.submit([&](sycl::handler &cgh) {
        auto acc = buf.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<TestKernel>(
            prg.get_kernel<TestKernel>(),
            [=]() SYCL_ESIMD_KERNEL { do_store(acc, i, spec_const.get()); });
      });
    } catch (cl::sycl::exception const &e) {
      std::cout << "SYCL exception caught: " << e.what() << '\n';
      return e.get_cl_code();
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
