// REQUIRES: gpu-amd-gfx90a
// RUN: %{build} -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental::matrix;

bool find_combination(const combination &comb,
                      const std::vector<combination> &expected_combinations) {
  return std::find_if(expected_combinations.begin(),
                      expected_combinations.end(),
                      [&comb](const auto &expected_comb) {
                        return (comb.max_msize == expected_comb.max_msize &&
                                comb.max_nsize == expected_comb.max_nsize &&
                                comb.max_ksize == expected_comb.max_ksize &&
                                comb.msize == expected_comb.msize &&
                                comb.nsize == expected_comb.nsize &&
                                comb.ksize == expected_comb.ksize &&
                                comb.atype == expected_comb.atype &&
                                comb.btype == expected_comb.btype &&
                                comb.ctype == expected_comb.ctype &&
                                comb.dtype == expected_comb.dtype);
                      }) != expected_combinations.end();
}

int main() {
  std::vector<combination> expected_combinations = {
      {0, 0, 0, 32, 32, 8, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 32, 32, 8, matrix_type::sint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {0, 0, 0, 16, 16, 16, matrix_type::sint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {0, 0, 0, 32, 32, 8, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 16, 16, 4, matrix_type::fp64, matrix_type::fp64,
       matrix_type::fp64, matrix_type::fp64}};

  sycl::queue q;
  std::vector<combination> actual_combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  assert(actual_combinations.size() == expected_combinations.size() &&
         "Number of combinations is not equal.");

  for (auto &comb : actual_combinations) {
    assert(find_combination(comb, expected_combinations) &&
           "Some values in matrix runtime query for gfx90a are not expected.");
  }
  return 0;
}
