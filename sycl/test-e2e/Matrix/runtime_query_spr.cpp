// REQUIRES: cpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental::matrix;

bool compare(const combination &lhs, const combination &rhs) {
  return (lhs.max_msize == rhs.max_msize && lhs.max_nsize == rhs.max_nsize &&
          lhs.max_ksize == rhs.max_ksize && lhs.msize == rhs.msize &&
          lhs.nsize == rhs.nsize && lhs.ksize == rhs.ksize &&
          lhs.atype == rhs.atype && lhs.btype == rhs.btype &&
          lhs.ctype == rhs.ctype && lhs.dtype == rhs.dtype);
}

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();
  if (!dev.ext_oneapi_architecture_is(
          sycl::ext::oneapi::experimental::architecture::intel_cpu_spr)) {
    std::cout << "Not an SPR architecture. Skipped." << std::endl;
    return 0;
  }

  std::vector<combination> expected_combinations = {
      {16, 16, 64, 0, 0, 0, matrix_type::uint8, matrix_type::uint8,
       matrix_type::sint32, matrix_type::sint32},
      {16, 16, 64, 0, 0, 0, matrix_type::uint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {16, 16, 64, 0, 0, 0, matrix_type::sint8, matrix_type::uint8,
       matrix_type::sint32, matrix_type::sint32},
      {16, 16, 64, 0, 0, 0, matrix_type::sint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {16, 16, 32, 0, 0, 0, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
  };

  std::vector<combination> actual_combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  assert(actual_combinations.size() == expected_combinations.size() &&
         "Number of combinations is not equal.");

  for (int i = 0; i < actual_combinations.size(); i++) {
    assert(compare(actual_combinations[i], expected_combinations[i]) &&
           "Some values in matrix runtime query for PVC are not expected.");
  }

  std::cout << "Passed." << std::endl;

  return 0;
}
