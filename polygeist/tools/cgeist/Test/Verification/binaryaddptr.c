// RUN: cgeist %s --function=* -S -o - | FileCheck %s
// XFAIL: *

#include <stdbool.h>

// COM: Currently not working for how operator+ is implemented for pointers.

bool f(bool *a, void *b) {
  return *a += b;
}
