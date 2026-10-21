// RUN: %{build} -fgpu-flush-denormals-to-zero -o %t.flush-denormals.out
// RUN: %{run} %t.flush-denormals.out

#include "common.hpp"
