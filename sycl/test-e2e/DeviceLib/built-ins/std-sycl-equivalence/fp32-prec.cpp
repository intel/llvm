// RUN: %{build} -foffload-fp32-prec-div -o %t.prec-div.out
// RUN: %{run} %t.prec-div.out
// RUN: %{build} -foffload-fp32-prec-sqrt -o %t.prec-sqrt.out
// RUN: %{run} %t.prec-sqrt.out

#include "common.hpp"
