// REQUIRES: matrix

// RUN: %{build} -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4
// RUN: %{run} %t.out

#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 16
// Sub-matrix N dimension
static constexpr size_t SN = 16;

#include "../joint_matrix_opt_kernel_feature_impl.hpp"