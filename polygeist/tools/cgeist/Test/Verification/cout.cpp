// RUN: cgeist %s --function=* -S | FileCheck %s
// XFAIL: *
#include <iostream>

void moo(int x) {
    std::cout << x << std::endl;
}
