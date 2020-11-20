// RUN: %clang_cc1 -fsycl -fsycl-is-host -triple spir64 -disable-llvm-passes %s -emit-llvm -o -  | FileCheck %s

class Functor16 {
public:
  [[intel::reqd_sub_group_size(16)]] void operator()() const {}
};

[[intel::reqd_sub_group_size(8)]] void foo() {}

class Functor {
public:
  void operator()() const {
    foo();
  }
};

template <int SIZE>
class Functor5 {
public:
  [[intel::reqd_sub_group_size(SIZE)]] void operator()() const {}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

void bar() {
  Functor16 f16;
  kernel<class kernel_name1>(f16);

  Functor f;
  kernel<class kernel_name2>(f);

  kernel<class kernel_name3>(
      []() [[intel::reqd_sub_group_size(4)]]{});

  Functor5<2> f5;
  kernel<class kernel_name4>(f5);
}

// CHECK: define spir_func void @_Z3foov() #0 !intel_reqd_sub_group_size ![[SGSIZE8:[0-9]+]]
// CHECK: define linkonce_odr spir_func void @_ZNK9Functor16clEv(%class.Functor16* %this) #2 comdat align 2 !intel_reqd_sub_group_size ![[SGSIZE16:[0-9]+]]
// CHECK: define internal spir_func void @"_ZZ3barvENK3$_0clEv"(%class.anon* %this) #2 align 2 !intel_reqd_sub_group_size ![[SGSIZE4:[0-9]+]]
// CHECK: define linkonce_odr spir_func void @_ZNK8Functor5ILi2EEclEv(%class.Functor5* %this) #2 comdat align 2 !intel_reqd_sub_group_size ![[SGSIZE2:[0-9]+]]
// CHECK: ![[SGSIZE8]] = !{i32 8}
// CHECK: ![[SGSIZE16]] = !{i32 16}
// CHECK: ![[SGSIZE4]] = !{i32 4}
// CHECK: ![[SGSIZE2]] = !{i32 2}
