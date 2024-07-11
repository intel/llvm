// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -fsycl-unique-prefix=THE_PREFIX -std=c++17 -fgpu-rdc -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-RDC
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -fsycl-unique-prefix=THE_PREFIX -std=c++17 -fno-gpu-rdc -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NORDC
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -fsycl-unique-prefix=THE_PREFIX -std=c++17 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOINTELFPGA
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -fsycl-unique-prefix=THE_PREFIX -std=c++17 -fintelfpga -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-INTELFPGA

#include "sycl.hpp"

// Test cases below show that 'sycl-unique-id' LLVM IR attribute is attached to the
// global variable whose type is decorated with device_global attribute, and that a
// unique string is generated.

using namespace sycl::ext::oneapi;
using namespace sycl;
queue q;

device_global<int> A;

[[intel::numbanks(2)]] device_global<int> Nonconst_glob;
[[intel::max_replicates(2)]] device_global<int> Nonconst_glob1;
[[intel::force_pow2_depth(1)]] device_global<int> Nonconst_glob2;
[[intel::bankwidth(2)]] device_global<int> Nonconst_glob3;
[[intel::simple_dual_port]] device_global<int> Nonconst_glob4;
[[intel::fpga_memory]] device_global<int> Nonconst_glob5;
[[intel::bank_bits(3, 4)]] device_global<int> Nonconst_glob6;
[[intel::fpga_register]] device_global<int> Nonconst_glob7;
[[intel::doublepump]] device_global<int>Nonconst_glob8;
[[intel::singlepump]] device_global<int> Nonconst_glob9;
[[intel::merge("mrg5", "width")]] device_global<int> Nonconst_glob10;
[[intel::private_copies(8)]] device_global<int> Nonconst_glob11;

#ifdef SYCL_EXTERNAL
SYCL_EXTERNAL device_global<int> AExt;
#endif
static device_global<int> B;

struct Foo {
  static device_global<int> C;
};
device_global<int> Foo::C;

// CHECK-RDC: @AExt = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[AEXT_ATTRS:[0-9]+]]
// CHECK: @A = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[A_ATTRS:[0-9]+]]
// CHECK: @Nonconst_glob = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[Non_Const_Num_ATTRS:[0-9]+]]
// CHECK: @Nonconst_glob1 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[Non_Const_Max_ATTRS:[0-9]+]]
// CHECK: @Nonconst_glob2 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[Non_Const_Force_ATTRS:[0-9]+]]
// CHECK: @Nonconst_glob3 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[Non_Const_Bankw_ATTRS:[0-9]+]]
// CHECK: @Nonconst_glob4 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[Non_Const_Simple_ATTRS:[0-9]+]]
// CHECK: @Nonconst_glob5 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[Non_Const_Mem_ATTRS:[0-9]+]]
// CHECK: @Nonconst_glob6 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[Non_Const_Bankbits_ATTRS:[0-9]+]]
// CHECK: @Nonconst_glob7 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[Non_Const_Reg_ATTRS:[0-9]+]]
// CHECK: @Nonconst_glob8 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[Non_Const_Dpump_ATTRS:[0-9]+]]
// CHECK: @Nonconst_glob9 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[Non_Const_Spump_ATTRS:[0-9]+]]
// CHECK: @Nonconst_glob10 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[Non_Const_Merge_ATTRS:[0-9]+]]
// CHECK: @Nonconst_glob11 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[Non_Const_Pc_ATTRS:[0-9]+]]
// CHECK: @_ZL1B = internal addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[B_ATTRS:[0-9]+]]
// CHECK: @_ZN3Foo1CE = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[C_ATTRS:[0-9]+]]

device_global<int> same_name;
namespace NS {
device_global<int> same_name;
}
// CHECK: @same_name = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[SAME_NAME_ATTRS:[0-9]+]]
// CHECK: @_ZN2NS9same_nameE = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[SAME_NAME_NS_ATTRS:[0-9]+]]

// decorated with only global_variable_allowed attribute
template <typename T>
class [[__sycl_detail__::global_variable_allowed]] only_global_var_allowed {
public:
  const T &get() const noexcept { return *Data; }
  only_global_var_allowed() {}
  operator T &() noexcept { return *Data; }

private:
  T *Data;
};

// check that we don't generate `sycl-unique-id` IR attribute if class does not use
// [[__sycl_detail__::device_global]]
only_global_var_allowed<int> no_device_global;
// CHECK-RDC: @no_device_global = linkonce_odr addrspace(1) global %class.only_global_var_allowed zeroinitializer, align 8{{$}}
// CHECK-NORDC: @no_device_global = internal addrspace(1) global %class.only_global_var_allowed zeroinitializer, align 8{{$}}

inline namespace Bar {
device_global<float> InlineNS;
}
// CHECK: @_ZN3Bar8InlineNSE = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.0" zeroinitializer, align 8 #[[BAR_INLINENS_ATTRS:[0-9]+]]

template <typename T> struct TS {
public:
  static device_global<T> d;
};
template <> device_global<int> TS<int>::d{};
// CHECK: @_ZN2TSIiE1dE = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[TEMPLATED_WRAPPER_ATTRS:[0-9]+]]

template <typename T>
device_global<T> templ_dev_global;
// CHECK: @[[TEMPL_DEV_GLOB:[a-zA-Z0-9_]+]] = linkonce_odr addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, comdat, align 8 #[[TEMPL_DEV_GLOB_ATTRS:[0-9]+]]

void foo() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_name_1>([=]() {
      (void)A;
      (void)Nonconst_glob;
      (void)Nonconst_glob1;
      (void)Nonconst_glob2;
      (void)Nonconst_glob3;
      (void)Nonconst_glob4;
      (void)Nonconst_glob5;
      (void)Nonconst_glob6;
      (void)Nonconst_glob7;
      (void)Nonconst_glob8;
      (void)Nonconst_glob9;
      (void)Nonconst_glob10;
      (void)Nonconst_glob11;
      (void)B;
      (void)Foo::C;
      (void)same_name;
      (void)NS::same_name;
      (void)no_device_global;
      (void)Bar::InlineNS;
      auto AA = TS<int>::d.get();
      auto val = templ_dev_global<int>.get();
    });
  });
}

namespace {
device_global<int> same_name;
}

// CHECK: [[ANN_numbanks:@.str]] = {{.*}}{memory:DEFAULT}{sizeinfo:8}{numbanks:2}{{.*}}
// CHECK: [[ANN_max_replicates:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:8}{max_replicates:2}{{.*}}
// CHECK: [[ANN_force_pow2_depth:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:8}{force_pow2_depth:1}{{.*}}
// CHECK: [[ANN_bankwidth:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:8}{bankwidth:2}{{.*}}
// CHECK: [[ANN_simple_dual_port:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:8}{simple_dual_port:1}{{.*}}
// CHECK: [[ANN_memory_default:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:8}{{.*}}
// CHECK: [[ANN_bank_bits:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:8}{numbanks:4}{bank_bits:3,4}{{.*}}
// CHECK: [[ANN_register:@.str[.0-9]*]] = {{.*}}{register:1}{{.*}}
// CHECK: [[ANN_doublepump:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:8}{pump:2}{{.*}}
// CHECK: [[ANN_singlepump:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:8}{pump:1}{{.*}}
// CHECK: [[ANN_merge:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:8}{merge:mrg5:width}{{.*}}
// CHECK: [[ANN_private_copies:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:8}{private_copies:8}{{.*}}
// CHECK: @counter = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.3" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS:[0-9]+]]
// CHECK: @counter1 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.4" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS1:[0-9]+]]
// CHECK: [[ANN_max_replicates1:@.str.[0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{max_replicates:2}{{.*}}
// CHECK: @counter2 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.5" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS2:[0-9]+]]
// CHECK: [[ANN_bankwidth1:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{bankwidth:2}{{.*}}
// CHECK: @counter3 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.6" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS3:[0-9]+]]
// CHECK: [[ANN_memory_default1:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{{.*}}
// CHECK: @counter4 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.7" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS4:[0-9]+]]
// CHECK: [[ANN_numbanks1:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{numbanks:2}{{.*}}
// CHECK: @counter5 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.8" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS5:[0-9]+]]
// CHECK: [[ANN_force_pow2_depth1:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{force_pow2_depth:1}{{.*}}
// CHECK: @counter6 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.9" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS6:[0-9]+]]
// CHECK: [[ANN_bank_bits1:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{numbanks:4}{bank_bits:3,4}{{.*}}
// CHECK: @counter7 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.10" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS7:[0-9]+]]
// CHECK: [[ANN_doublepump1:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{pump:2}{{.*}}
// CHECK: @counter8 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.11" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS8:[0-9]+]]
// CHECK: [[ANN_singlepump1:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{pump:1}{{.*}}
// CHECK: @counter9 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.12" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS9:[0-9]+]]
// CHECK: [[ANN_merge1:@.str[.0-9]*]]  = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{merge:mrg5:width}{{.*}}
// CHECK: @counter10 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.13" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS10:[0-9]+]]
// CHECK: [[ANN_private_copies1:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{private_copies:8}{{.*}}
// CHECK: @counter11 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.14" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS11:[0-9]+]]
// CHECK: [[ANN_simple_dual_port1:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{simple_dual_port:1}{{.*}}
// CHECK: @counter12 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.15" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS12:[0-9]+]]
// CHECK: [[ANN_simple_dual_port2:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{simple_dual_port:1}{{.*}}
// CHECK: [[ANN_private_copies2:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{private_copies:16}{{.*}}
// CHECK: @counter13 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.16" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS13:[0-9]+]]
// CHECK: [[ANN_numbanks2:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{numbanks:2}{{.*}}
// CHECK: [[ANN_merge2:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{merge:foo:depth}{{.*}}
// CHECK: @counter14 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.17" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS14:[0-9]+]]
// CHECK: [[ANN_force_pow2_depth2:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{force_pow2_depth:0}{{.*}}
// CHECK: [[ANN_doublepump2:@.str[.0-9]*]] = {{.*}}"{memory:DEFAULT}{sizeinfo:4}{pump:2}{{.*}}
// CHECK: [[ANN_bank_bits2:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,155}{numbanks:4}{bank_bits:2,3}{{.*}}
// CHECK: @counter15 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.18" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS15:[0-9]+]]
// CHECK-INTELFPGA: @global_wrapper = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.19" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS16:[0-9]+]]
// CHECK-INTELFPGA: [[ANN_memory3:@.str[.0-9]*]] = {{.*}}{memory:MLAB}{sizeinfo:48}{{.*}}
// CHECK-INTELFPGA: [[ANN_numbanks4:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,10}{numbanks:4}{{.*}}
// CHECK-INTELFPGA: @global_wrapper1 = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.20" zeroinitializer, align 8 #[[DEV_GLOB_FPGA_ATTRS17:[0-9]+]]
// CHECK-INTELFPGA: [[ANN_private_copies4:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:48}{private_copies:16}{{.*}}
// CHECK-INTELFPGA: [[ANN_simple_dual_port4:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4,10}{simple_dual_port:1}{{.*}}
// CHECK-INTELFPGA: [[ANN_private_copies_imple_dual_port2:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{simple_dual_port:1}{memory:DEFAULT}{sizeinfo:4}{private_copies:16}{{.*}}
// CHECK-INTELFPGA: [[ANN_numbanks3_merge3:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:4}{numbanks:2}{memory:DEFAULT}{sizeinfo:4,155}{merge:foo:depth}{{.*}}
// CHECK-INTELFPGA: [[ANN_force_pow2_depth3_doublepump3_numbanks3:@.str[.0-9]*]]  = {{.*}}{memory:DEFAULT}{sizeinfo:4}{force_pow2_depth:0}{memory:DEFAULT}{sizeinfo:4}{pump:2}{memory:DEFAULT}{sizeinfo:4,155}{numbanks:4}{bank_bits:2,3}{{.*}}
// CHECK-INTELFPGA: [[ANN_memory4_numbanks4_register4:@.str[.0-9]*]] = {{.*}}{memory:MLAB}{sizeinfo:48}{memory:DEFAULT}{sizeinfo:4,10}{numbanks:4}{register:1}{register:1}{{.*}}
// CHECK-INTELFPGA: [[ANN_private_copies5_simple_dual_port5_register5:@.str[.0-9]*]] = {{.*}}{memory:DEFAULT}{sizeinfo:48}{private_copies:16}{memory:DEFAULT}{sizeinfo:4,10}{simple_dual_port:1}{register:1}{{.*}}
// CHECK: @_ZN12_GLOBAL__N_19same_nameE = internal addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[SAME_NAME_ANON_NS_ATTRS:[0-9]+]]

struct bar {
  int t1;
  int t2;
  [[intel::fpga_register]] int arr[155];
};

device_global<bar> counter;

struct bar1 {
  int t3;
  int t4;
  [[intel::max_replicates(2)]] int arr1[155];
};

device_global<bar1> counter1;

struct bar2 {
  int t5;
  int t6;
  [[intel::bankwidth(2)]] int arr2[155];
};

device_global<bar2> counter2;

struct bar3 {
  int t7;
  int t8;
  [[intel::fpga_memory]] int arr3[155];
};

device_global<bar3> counter3;

struct bar4 {
  int t9;
  int t10;
  [[intel::numbanks(2)]] int arr4[155];
};

device_global<bar4> counter4;

struct bar5 {
  int t11;
  int t12;
  [[intel::force_pow2_depth(1)]] int arr5[155];
};

device_global<bar5> counter5;

struct bar6 {
  int t13;
  int t14;
  [[intel::bank_bits(3, 4)]] int arr6[155];
};

device_global<bar6> counter6;

struct bar7 {
  int t15;
  int t16;
  [[intel::doublepump]] int arr7[155];
};

device_global<bar7> counter7;

struct bar8 {
  int t17;
  int t18;
  [[intel::singlepump]] int arr8[155];
};

device_global<bar8> counter8;

struct bar9 {
  int t19;
  int t20;
  [[intel::merge("mrg5", "width")]] int arr9[155];
};

device_global<bar9> counter9;

struct bar10 {
  int t21;
  int t22;
  [[intel::private_copies(8)]] int arr10[155];
};

device_global<bar10> counter10;

struct bar11 {
  int t23;
  int t24;
  [[intel::simple_dual_port]] int arr11[155];
};

device_global<bar11> counter11;

struct bar12 {
  [[intel::simple_dual_port]] int t25;
  [[intel::private_copies(16)]] int t26;
};

device_global<bar12> counter12;

struct bar13 {
  [[intel::numbanks(2)]] int t27;
  [[intel::merge("foo", "depth")]] int arr12[155];
};

device_global<bar13> counter13;

struct bar14 {
  [[intel::force_pow2_depth(0)]] int t28;
  [[intel::doublepump]] int t29;
  [[intel::bank_bits(2, 3)]] int arr13[155];
};

device_global<bar14> counter14;

struct bar15 {
  int t30;
  int t31;
  int arr14[155];
};

[[intel::fpga_register]] device_global<bar15> counter15;

// Base class with different attributes
class Base {
public:
  [[intel::fpga_register]] int reg_attr;
  int no_attr;
};

// Derived class with additional attributes
class Derived : public Base {
public:
  [[intel::numbanks(4)]] int arr_attr[10];
};

// Class with class type member with attributes
class Wrapper {
public:
  [[intel::fpga_memory("MLAB")]] Derived derived_attr;
};

// Global instance with FPGA attributes
[[intel::fpga_register]] device_global<Wrapper> global_wrapper;

// Base class with different attributes
class Base1 {
public:
  [[intel::fpga_register]] int reg_attr1;
  int no_attr1;
};

// Derived class with additional attributes
class Derived1 : public Base1 {
public:
  [[intel::simple_dual_port]] int arr_attr1[10];
};

// Class with class type member with attributes
class Wrapper1 {
public:
  [[intel::private_copies(16)]] Derived1 derived_attr1;
};

// Global instance with FPGA attributes
device_global<Wrapper1> global_wrapper1;


int main() {
  queue q;

  q.submit([&](handler &h) {
    h.single_task<class kernel_name_2>([=] {

    auto& non_const_counter = const_cast<bar&>(counter.get());
    non_const_counter.arr[0]++;

    auto& non_const_counter1 = const_cast<bar1&>(counter1.get());
    non_const_counter1.arr1[0]++;

    auto& non_const_counter2 = const_cast<bar2&>(counter2.get());
    non_const_counter2.arr2[0]++;

    auto& non_const_counter3 = const_cast<bar3&>(counter3.get());
    non_const_counter3.arr3[0]++;

    auto& non_const_counter4 = const_cast<bar4&>(counter4.get());
    non_const_counter4.arr4[0]++;

    auto& non_const_counter5 = const_cast<bar5&>(counter5.get());
    non_const_counter5.arr5[0]++;

    auto& non_const_counter6 = const_cast<bar6&>(counter6.get());
    non_const_counter6.arr6[0]++;

    auto& non_const_counter7 = const_cast<bar7&>(counter7.get());
    non_const_counter7.arr7[0]++;

    auto& non_const_counter8 = const_cast<bar8&>(counter8.get());
    non_const_counter8.arr8[0]++;

    auto& non_const_counter9 = const_cast<bar9&>(counter9.get());
    non_const_counter9.arr9[0]++;

    auto& non_const_counter10 = const_cast<bar10&>(counter10.get());
    non_const_counter10.arr10[0]++;

    auto& non_const_counter11 = const_cast<bar11&>(counter11.get());
    non_const_counter11.arr11[0]++;

    auto& non_const_counter12 = const_cast<bar12&>(counter12.get());
    non_const_counter12.t25 = 5;
    non_const_counter12.t26 = 20;

    auto& non_const_counter13 = const_cast<bar13&>(counter13.get());
    non_const_counter13.t27 = 30;
    non_const_counter13.arr12[0]++;

    auto& non_const_counter14 = const_cast<bar14&>(counter14.get());
    non_const_counter14.t28 = 35;
    non_const_counter14.t29 = 36;
    non_const_counter14.arr13[0]++;

    auto& non_const_counter15 = const_cast<bar15&>(counter15.get());
    non_const_counter15.t30 = 5;
    non_const_counter15.t31 = 9;
    non_const_counter15.arr14[0]++;

    auto& non_const_global_wrapper = const_cast<Wrapper&>(global_wrapper.get());
    non_const_global_wrapper.derived_attr.reg_attr = 5;
    non_const_global_wrapper.derived_attr.no_attr = 10;
    for (int i = 0; i < 10; ++i) {
      non_const_global_wrapper.derived_attr.arr_attr[i] = i;
    }

    auto& non_const_global_wrapper1 = const_cast<Wrapper1&>(global_wrapper1.get());
    non_const_global_wrapper1.derived_attr1.reg_attr1 = 3;
    non_const_global_wrapper1.derived_attr1.no_attr1 = 20;
    for (int i = 0; i < 10; ++i) {
      non_const_global_wrapper1.derived_attr1.arr_attr1[i] = i;
    }

    });
  });

  q.wait();
  return 0;
}

namespace {
void bar() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_name>([=]() { int A = same_name; });
  });
}
} // namespace

// CHECK: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr addrspace(4) }] [{ i32, ptr, ptr addrspace(4) } { i32 65535, ptr @__cxx_global_var_init{{.*}}, ptr addrspace(4) addrspacecast (ptr addrspace(1) @[[TEMPL_DEV_GLOB]] to ptr addrspace(4)) }, { i32, ptr, ptr addrspace(4) } { i32 65535, ptr @_GLOBAL__sub_I_device_global.cpp, ptr addrspace(4) null }]

// CHECK: @llvm.global.annotations
// CHECK-NOINTELFPGA-SAME: ptr addrspace(1) @Nonconst_glob,  ptr addrspace(1) [[ANN_numbanks]]{{.*}} i32 18, ptr addrspace(1) null
// CHECK-NOINTELFPGA-SAME: ptr addrspace(1) @Nonconst_glob1, ptr addrspace(1) [[ANN_max_replicates]]{{.*}} i32 19, ptr addrspace(1) null
// CHECK-NOINTELFPGA-SAME: ptr addrspace(1) @Nonconst_glob2, ptr addrspace(1) [[ANN_force_pow2_depth]]{{.*}} i32 20, ptr addrspace(1) null
// CHECK-NOINTELFPGA-SAME: ptr addrspace(1) @Nonconst_glob3, ptr addrspace(1) [[ANN_bankwidth]]{{.*}} i32 21, ptr addrspace(1) null
// CHECK-NOINTELFPGA-SAME: ptr addrspace(1) @Nonconst_glob4, ptr addrspace(1) [[ANN_simple_dual_port]]{{.*}} i32 22, ptr addrspace(1) null
// CHECK-NOINTELFPGA-SAME: ptr addrspace(1) @Nonconst_glob5, ptr addrspace(1) [[ANN_memory_default]]{{.*}} i32 23, ptr addrspace(1) null
// CHECK-NOINTELFPGA-SAME: ptr addrspace(1) @Nonconst_glob6, ptr addrspace(1) [[ANN_bank_bits]]{{.*}} i32 24, ptr addrspace(1) null
// CHECK-NOINTELFPGA-SAME: ptr addrspace(1) @Nonconst_glob7, ptr addrspace(1) [[ANN_register]]{{.*}} i32 25, ptr addrspace(1) null
// CHECK-NOINTELFPGA-SAME: ptr addrspace(1) @Nonconst_glob8, ptr addrspace(1) [[ANN_doublepump]]{{.*}} i32 26, ptr addrspace(1) null
// CHECK-NOINTELFPGA-SAME: ptr addrspace(1) @Nonconst_glob9, ptr addrspace(1) [[ANN_singlepump]]{{.*}} i32 27, ptr addrspace(1) null
// CHECK-NOINTELFPGA-SAME: ptr addrspace(1) @Nonconst_glob10, ptr addrspace(1) [[ANN_merge]]{{.*}} i32 28, ptr addrspace(1) null
// CHECK-NOINTELFPGA-SAME: ptr addrspace(1) @Nonconst_glob11, ptr addrspace(1) [[ANN_private_copies]]{{.*}} i32 29, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter, ptr addrspace(1) [[ANN_register]]{{.*}}, i32 196, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter1, ptr addrspace(1) [[ANN_max_replicates1]]{{.*}}, i32 204, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter2, ptr addrspace(1) [[ANN_bankwidth1]]{{.*}}, i32 212, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter3, ptr addrspace(1) [[ANN_memory_default1]]{{.*}}, i32 220, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter4, ptr addrspace(1) [[ANN_numbanks1]]{{.*}}, i32 228, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter5, ptr addrspace(1) [[ANN_force_pow2_depth1]]{{.*}}, i32 236, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter6, ptr addrspace(1) [[ANN_bank_bits1]]{{.*}}, i32 244, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter7, ptr addrspace(1) [[ANN_doublepump1]]{{.*}}, i32 252, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter8, ptr addrspace(1) [[ANN_singlepump1]]{{.*}}, i32 260, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter9, ptr addrspace(1) [[ANN_merge1]]{{.*}}, i32 268, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter10, ptr addrspace(1) [[ANN_private_copies1]]{{.*}}, i32 276, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter11, ptr addrspace(1) [[ANN_simple_dual_port1]]{{.*}}, i32 284, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter12, ptr addrspace(1) [[ANN_private_copies_imple_dual_port2]]{{.*}}, i32 291, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter13, ptr addrspace(1) [[ANN_numbanks3_merge3]]{{.*}}, i32 298, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter14, ptr addrspace(1) [[ANN_force_pow2_depth3_doublepump3_numbanks3]]{{.*}}, i32 306, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @counter15, ptr addrspace(1) [[ANN_register]]{{.*}}, i32 314, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @global_wrapper, ptr addrspace(1) [[ANN_memory4_numbanks4_register4]]{{.*}}i32 336, ptr addrspace(1) null
// CHECK-INTELFPGA-SAME: ptr addrspace(1) @global_wrapper1, ptr addrspace(1) [[ANN_private_copies5_simple_dual_port5_register5]]{{.*}}i32 358, ptr addrspace(1) null
// CHECK: @llvm.used = appending addrspace(1) global [1 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @[[TEMPL_DEV_GLOB]] to ptr addrspace(4))], section "llvm.metadata"
// CHECK: @llvm.compiler.used = appending addrspace(1) global [2 x ptr addrspace(4)]
// CHECK-SAME: @_ZL1B
// CHECK-SAME: @_ZN12_GLOBAL__N_19same_nameE

// CHECK-RDC: attributes #[[AEXT_ATTRS]] = { "sycl-unique-id"="_Z4AExt" }
// CHECK: attributes #[[A_ATTRS]] = { "sycl-unique-id"="_Z1A" }
// CHECK: attributes #[[Non_Const_Num_ATTRS]] = { "sycl-unique-id"="_Z13Nonconst_glob" }
// CHECK: attributes #[[Non_Const_Max_ATTRS]] = { "sycl-unique-id"="_Z14Nonconst_glob1" }
// CHECK: attributes #[[Non_Const_Force_ATTRS]] = { "sycl-unique-id"="_Z14Nonconst_glob2" }
// CHECK: attributes #[[Non_Const_Bankw_ATTRS]] = { "sycl-unique-id"="_Z14Nonconst_glob3" }
// CHECK: attributes #[[Non_Const_Simple_ATTRS]] = { "sycl-unique-id"="_Z14Nonconst_glob4" }
// CHECK: attributes #[[Non_Const_Mem_ATTRS]] = { "sycl-unique-id"="_Z14Nonconst_glob5" }
// CHECK: attributes #[[Non_Const_Bankbits_ATTRS]] = { "sycl-unique-id"="_Z14Nonconst_glob6" }
// CHECK: attributes #[[Non_Const_Reg_ATTRS]] = { "sycl-unique-id"="_Z14Nonconst_glob7" }
// CHECK: attributes #[[Non_Const_Dpump_ATTRS]] = { "sycl-unique-id"="_Z14Nonconst_glob8" }
// CHECK: attributes #[[Non_Const_Spump_ATTRS]] = { "sycl-unique-id"="_Z14Nonconst_glob9" }
// CHECK: attributes #[[Non_Const_Merge_ATTRS]] = { "sycl-unique-id"="_Z15Nonconst_glob10" }
// CHECK: attributes #[[Non_Const_Pc_ATTRS]] = { "sycl-unique-id"="_Z15Nonconst_glob11" }
// CHECK: attributes #[[B_ATTRS]] = { "sycl-unique-id"="THE_PREFIX____ZL1B" }
// CHECK: attributes #[[C_ATTRS]] = { "sycl-unique-id"="_ZN3Foo1CE" }
// CHECK: attributes #[[SAME_NAME_ATTRS]] = { "sycl-unique-id"="_Z9same_name" }
// CHECK: attributes #[[SAME_NAME_NS_ATTRS]] = { "sycl-unique-id"="_ZN2NS9same_nameE" }
// CHECK: attributes #[[BAR_INLINENS_ATTRS]] = { "sycl-unique-id"="_ZN3Bar8InlineNSE" }
// CHECK: attributes #[[TEMPLATED_WRAPPER_ATTRS]] = { "sycl-unique-id"="_ZN2TSIiE1dE" }
// CHECK: attributes #[[TEMPL_DEV_GLOB_ATTRS]] = { "sycl-unique-id"="_Z16templ_dev_globalIiE" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS]] = { "sycl-unique-id"="_Z7counter" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS1]] = { "sycl-unique-id"="_Z8counter1" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS2]] = { "sycl-unique-id"="_Z8counter2" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS3]] = { "sycl-unique-id"="_Z8counter3" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS4]] = { "sycl-unique-id"="_Z8counter4" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS5]] = { "sycl-unique-id"="_Z8counter5" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS6]] = { "sycl-unique-id"="_Z8counter6" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS7]] = { "sycl-unique-id"="_Z8counter7" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS8]] = { "sycl-unique-id"="_Z8counter8" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS9]] = { "sycl-unique-id"="_Z8counter9" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS10]] = { "sycl-unique-id"="_Z9counter10" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS11]] = { "sycl-unique-id"="_Z9counter11" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS12]] = { "sycl-unique-id"="_Z9counter12" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS13]] = { "sycl-unique-id"="_Z9counter13" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS14]] = { "sycl-unique-id"="_Z9counter14" }
// CHECK: attributes #[[DEV_GLOB_FPGA_ATTRS15]] = { "sycl-unique-id"="_Z9counter15" }
// CHECK-INTELFPGA: attributes #[[DEV_GLOB_FPGA_ATTRS16]] = { "sycl-unique-id"="_Z14global_wrapper" }
// CHECK-INTELFPGA: attributes #[[DEV_GLOB_FPGA_ATTRS17]] = { "sycl-unique-id"="_Z15global_wrapper1" }

// CHECK: attributes #[[SAME_NAME_ANON_NS_ATTRS]] = { "sycl-unique-id"="THE_PREFIX____ZN12_GLOBAL__N_19same_nameE" }
