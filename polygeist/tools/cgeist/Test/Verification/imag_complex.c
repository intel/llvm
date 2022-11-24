// RUN: cgeist %s --function=* -S | FileCheck %s
// XFAIL: *

#include <complex.h>

int f2(int complex a) {
  return __imag__(a);
}

float f3(float complex a) {
  return __imag__(a);
}
