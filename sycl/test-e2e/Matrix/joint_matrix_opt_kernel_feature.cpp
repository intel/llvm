// REQUIRES: matrix

// RUN: %{build} -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4
// RUN: %{run} %t.out

// Test checks that exception will be thrown in case matrix parameters are
// incompatible on the current device

#include "common.hpp"

#define SG_SZ 16
static constexpr size_t SN = 16;

#include "joint_matrix_opt_kernel_feature_impl.hpp"
