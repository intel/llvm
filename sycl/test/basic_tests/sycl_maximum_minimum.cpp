// RUN: %clangxx -fsycl -DNDEBUG %s -o %t.out
// RUN: %t.out
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <cassert>
#include <iostream>
#include <CL/sycl.hpp>

#include <algorithm>
#include <limits>

int main() {
    double f1 = 1.0f;
    double f2 = std::numeric_limits<double>::quiet_NaN();

    // std::cout << std::max(f1,f2) << ", " << sycl::maximum{}(f1,f2) << "\n"; //--> result:1
    // std::cout << std::max(f2,f1) << ", " << sycl::maximum{}(f2,f1) << "\n"; //--> result: nan  
    // std::cout << std::min(f1,f2) << ", " << sycl::minimum{}(f1,f2) << "\n"; //--> result: 1
    // std::cout << std::min(f2,f1) << ", " << sycl::minimum{}(f2,f1) << "\n"; //--> result: nan

    assert(((std::max(f1,f2) == sycl::maximum{}(f1,f2)) && "sycl::maximum result is wrong"));
    //assert(((double)((std::max(f2,f1))) == (double)(sycl::maximum{}(f2,f1))) && "sycl::maximum result is wrong"); //==> differnet nan type
    assert(((std::min(f1,f2) == sycl::minimum{}(f1,f2)) && "sycl::minimum result is wrong"));
    //assert(((double)((std::min(f2,f1))) == (double)(sycl::minimum{}(f1,f2))) && "sycl::minimum result is wrong"); //==> differnet nan type

    return 0;    
}
