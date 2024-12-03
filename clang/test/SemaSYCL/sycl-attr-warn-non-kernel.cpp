// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -verify -pedantic %s

// The test check issuing diagnostics for attributes that can not be applied to a non SYCL kernel function

#include "sycl.hpp" //clang/test/SemaSYCL/Inputs/sycl.hpp

[[sycl::reqd_work_group_size(16)]] void f1(){ // expected-warning {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
}

[[sycl::reqd_sub_group_size(12)]] void f3(){ // expected-warning {{'reqd_sub_group_size' attribute can only be applied to a SYCL kernel function}}
}

[[sycl::reqd_work_group_size(16)]] void f4(){ // expected-warning {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
}

[[sycl::work_group_size_hint(8, 8, 8)]] void f5(){}; // expected-warning {{'work_group_size_hint' attribute can only be applied to a SYCL kernel function}}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(16)]] void f6() {} // expected-warning {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}

class Functor16 {
public:
  [[sycl::reqd_work_group_size(16)]] void operator()() const;
};
void Functor16::operator()() const {
}

class Functor16x16 {
public:
  [[sycl::reqd_work_group_size(16,16)]] void operator()() const; // expected-warning {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
};
void Functor16x16::operator()() const {
}


class Functor16x16x16 {
public:
  [[sycl::reqd_work_group_size(16,16,16)]] void operator()() const{

  } 
};

class FunctorSubGroupSize4 {
public:
  [[sycl::reqd_sub_group_size(4)]] void operator()() const{}  // expected-warning {{'reqd_sub_group_size' attribute can only be applied to a SYCL kernel function}}
};

class Functor8 {
public:
  void operator()() const;  
};

[[sycl::reqd_work_group_size(8)]] void Functor8::operator()() const {} // expected-warning {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}

int main() {
  sycl::queue q;
  Functor16x16 f16x16;
  FunctorSubGroupSize4 fs4;
  Functor8 f8;
  
  q.submit([&](sycl::handler& h) {
    h.single_task<class kernel_name>(
      []()[[sycl::reqd_work_group_size(16)]]{ // OK attribute reqd_work_group_size applied to kernel
        f1();
      } 
    );
  });


  q.submit([&](sycl::handler &h) {
    Functor16 f16;
    Functor16x16x16 f16x16x16;
    h.single_task<class kernel_name1>(f16);  // OK attribute reqd_work_group_size applied to kernel
    h.single_task<class kernel_name2>(f16x16x16); // OK attribute reqd_work_group_size applied to kernel
  });


  return 0;
}
