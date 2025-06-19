// This is not meant as a standalone test but rather as a source file that will
// link with SeparateCompilation.cpp to check that separate compilation works
// with free function kernels. Hence the .cc suffix to exclude it from the list
// of picked up tests.

#include "SumKernel.hpp"
#include <sycl/ext/oneapi/free_function_queries.hpp>

using namespace sycl;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void SumKernel::sum(accessor<int, 1> accA, accessor<int, 1> accB,
                    accessor<int, 1> result) {
  size_t id =
      ext::oneapi::this_work_item::get_nd_item<1>().get_global_linear_id();
  result[id] = accA[id] + accB[id];
}
