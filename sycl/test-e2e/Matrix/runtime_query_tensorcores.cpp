// REQUIRES: cuda
// RUN: %{build} -o %t.out
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

  sycl::queue Q;
  auto ComputeCapability =
      std::stof(Q.get_device().get_info<sycl::info::device::backend_version>());

  std::vector<combination> sm_70_combinations = {
      {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 8, 32, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 32, 8, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp16, matrix_type::fp16},
      {0, 0, 0, 8, 32, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp16, matrix_type::fp16},
      {0, 0, 0, 32, 8, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp16, matrix_type::fp16},
      {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp16},
      {0, 0, 0, 8, 32, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp16},
      {0, 0, 0, 32, 8, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp32, matrix_type::fp16},
      {0, 0, 0, 16, 16, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp16, matrix_type::fp32},
      {0, 0, 0, 8, 32, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp16, matrix_type::fp32},
      {0, 0, 0, 32, 8, 16, matrix_type::fp16, matrix_type::fp16,
       matrix_type::fp16, matrix_type::fp32}};

  std::vector<combination> sm_72_combinations = {
      {0, 0, 0, 16, 16, 16, matrix_type::sint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {0, 0, 0, 8, 32, 16, matrix_type::sint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {0, 0, 0, 32, 8, 16, matrix_type::sint8, matrix_type::sint8,
       matrix_type::sint32, matrix_type::sint32},
      {0, 0, 0, 16, 16, 16, matrix_type::uint8, matrix_type::uint8,
       matrix_type::sint32, matrix_type::sint32},
      {0, 0, 0, 8, 32, 16, matrix_type::uint8, matrix_type::uint8,
       matrix_type::sint32, matrix_type::sint32},
      {0, 0, 0, 32, 8, 16, matrix_type::uint8, matrix_type::uint8,
       matrix_type::sint32, matrix_type::sint32}};

  std::vector<combination> sm_80_combinations = {
      {0, 0, 0, 16, 16, 8, matrix_type::tf32, matrix_type::tf32,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 16, 16, 16, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 8, 32, 16, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 32, 8, 16, matrix_type::bf16, matrix_type::bf16,
       matrix_type::fp32, matrix_type::fp32},
      {0, 0, 0, 8, 8, 4, matrix_type::fp64, matrix_type::fp64,
       matrix_type::fp64, matrix_type::fp64}};

  std::vector<combination> expected_combinations;

  if (ComputeCapability >= 8.0) {
    std::move(sm_70_combinations.begin(), sm_70_combinations.end(),
              std::back_inserter(expected_combinations));
    std::move(sm_72_combinations.begin(), sm_72_combinations.end(),
              std::back_inserter(expected_combinations));
    std::move(sm_80_combinations.begin(), sm_80_combinations.end(),
              std::back_inserter(expected_combinations));
  } else if (ComputeCapability >= 7.2) {
    std::move(sm_70_combinations.begin(), sm_70_combinations.end(),
              std::back_inserter(expected_combinations));
    std::move(sm_72_combinations.begin(), sm_72_combinations.end(),
              std::back_inserter(expected_combinations));
  } else if (ComputeCapability >= 7.0) {
    std::move(sm_70_combinations.begin(), sm_70_combinations.end(),
              std::back_inserter(expected_combinations));
  } else {
    return 0;
  }

  std::vector<combination> actual_combinations =
      Q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  assert(actual_combinations.size() == expected_combinations.size() &&
         "Number of combinations is not equal.");

  for (auto &comb : actual_combinations) {
    assert(find_combination(comb, expected_combinations) &&
           "Some values in matrix runtime query for CUDA are not expected.");
  }

  return 0;
}
