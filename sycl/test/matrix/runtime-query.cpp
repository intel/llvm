// RUN: %clangxx -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -fsycl -o runtime-query %s
// XFAIL: *

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;

template <matrix_type Ta, matrix_type Tb, matrix_type Tc, matrix_type Td>
void matrix_runtime_query(queue q) {

  std::vector<combination> combinations =
      q.get_device().get_info<sycl::info::device::matrix_combinations>();

  std::cout << "The matrix hardware implementation in this device provides "
               "this number of combinations: "
            << combinations.size() << std::endl;

  bool max_sizes;
  if (combinations[0].maxsize == 0)
    max_sizes = true; // this is a max params hardware
  else
    max_sizes = false;
  for (int i = 0; i < combinations.size(); i++) {
    if (Ta == combinations[i].atype && Tb == combinations[i].btype &&
        Tc == combinations[i].ctype && Td == combinations[i].dtype) {
      // joint matrix GEMM kernel can be called using these sizes
      if (max_sizes)
        std::cout << "The matrix hardware implementation in this device "
                     "provides the following max sizes are: M "
                  << combinations[i].max_msize << " N "
                  << combinations[i].max_nsize << " K "
                  << combinations[i].max_ksize << std::endl;
      else
        std::cout << "The matrix hardware implementation in this device "
                     "provides the following sizes are: M "
                  << combinations[i].msize << " N " << combinations[i].nsize
                  << " K " << combinations[i].ksize << std::endl;
    }
  }
}

int main() {
  queue q;
  matrix_runtime_query<matrix_type::bf16, matrix_type::bf16, matrix_type::float,
                       matrix_type::float>(q);
  return 0;
}
