//RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl -fsycl-is-device -triple spir64 -Wno-sycl-2017-compat -verify | FileCheck %s

#include "Inputs/sycl.hpp"
/*
int main() {
    cl::sycl::kernel_single_task<class test_kernel1>(
            []() {});

    cl::sycl::kernel_single_task([]() {});
}
*/
template<typename Func1, typename Func2>
void calls_kernel(Func1 f1, Func2 f2) {
    cl::sycl::kernel_single_task<class Foo>([](){/*kernel lambda*/});
}

int main() {
                calls_kernel([](int i){return i;}, [](double d){return d;});
}

