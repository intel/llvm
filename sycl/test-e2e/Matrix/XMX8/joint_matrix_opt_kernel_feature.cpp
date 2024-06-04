// REQUIRES: matrix-xmx8

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test checks that exception will be thrown in case matrix parameters are
// incompatible on the current device

#include "../common.hpp"
#include "../joint_matrix_opt_kernel_feature_impl.hpp"
