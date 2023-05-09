// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include "add.h"

int main() { add_test_all<access::address_space::global_space>(); }
