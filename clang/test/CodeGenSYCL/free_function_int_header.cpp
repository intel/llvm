// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s
// 
// This test checks integration header contents for free functions with scalar,
// pointer, non-decomposed struct parameters, work group memory parameters,
// dynamic work group memory parameters and special types.

#include "mock_properties.hpp"
#include "sycl.hpp"

// First overload of function ff_2.
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
                                              2)]] void
ff_2(int *ptr, int start, int end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start + 66;
}

// Second overload of function ff_2.
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
  2)]] void
  ff_2(int* ptr, int start, int end, int value) {
  for (int i = start; i <= end; i++)
    ptr[i] = start + value;
}

// Templated definition of function ff_3.
template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
ff_3(T *ptr, T start, T end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start;
}

// Explicit instantiation of ff_3 with int type.
template void ff_3(int *ptr, int start, int end);

// Explicit instantiation of ff_3 with float type.
template void ff_3(float* ptr, float start, float end);

// Specialization of ff_3 with double type.
template <> void ff_3<double>(double *ptr, double start, double end) {
  for (int i = start; i <= end; i++)
    ptr[i] = end;
}

struct NoPointers {
  int f;
};

struct Pointers {
  int * a;
  float * b;
};

struct Agg {
  NoPointers F1;
  int F2;
  int *F3;
  Pointers F4;
};

struct Derived : Agg {
  int a;
};

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_4(NoPointers S1, Pointers S2, Agg S3) {
}

template <typename T1, typename T2>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
  void ff_6(T1 S1, T2 S2, int end) {
}

template void ff_6(Agg S1, Derived S2, int);

constexpr int TestArrSize = 3;
constexpr int TestArrSizeAlias = 50;

template <int ArrSize>
struct KArgWithPtrArray {
  int *data[ArrSize];
  int start[ArrSize];
  int end[ArrSize];
  constexpr int getArrSize() { return ArrSize; }
};

namespace free_functions {
  template <int ArrSize>
  struct KArgWithPtrArray {
    float *data[ArrSize];
    float start[ArrSize];
    float end[ArrSize];
    constexpr int getArrSize() { return ArrSize; }
  };

  using AliasStruct = KArgWithPtrArray<TestArrSizeAlias>;
}

template <int ArrSize>
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_7(KArgWithPtrArray<ArrSize> KArg) {
  for (int j = 0; j < ArrSize; j++)
    for (int i = KArg.start[j]; i <= KArg.end[j]; i++)
      KArg.data[j][i] = KArg.start[j] + KArg.end[j];
}

template void ff_7(KArgWithPtrArray<TestArrSize> KArg);

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]] 
void ff_8(sycl::work_group_memory<int>) {
}

// function in namespace
namespace free_functions {
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_9(int start, int *ptr) {
}
}

// function in nested namespace
namespace free_functions::tests {
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_10(int start, int *ptr) {
}
}

// function in inline namespace
namespace free_functions::tests {
inline namespace V1 {
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_11(int start, int *ptr) {
}
}
}

//function in anonymous namespace
namespace {
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_12(int start, int *ptr) {
}
}

// functions with the same name but in different namespaces
namespace free_functions {
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_13(int start, int *ptr) {
}
}
namespace free_functions::tests {
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_13(int start, int *ptr) {
}
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]] 
void ff_9(sycl::dynamic_work_group_memory<int>) {
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_11(sycl::local_accessor<int, 1> lacc) {
}

template <typename DataT>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_11(sycl::local_accessor<DataT, 1> lacc) {
}

template void ff_11(sycl::local_accessor<float, 1> lacc);

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_12(sycl::sampler S) {
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_13(sycl::stream str) {
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_14(sycl::ext::oneapi::experimental::annotated_arg<int> arg) {
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_15(sycl::ext::oneapi::experimental::annotated_ptr<int> ptr) {
}

typedef int TypedefType;
using AliasType = Derived;

namespace free_functions::tests {
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_14(TypedefType start, TypedefType *ptr) {
}
}

namespace free_functions::tests {
typedef int NamespaceTypedefType;
using AliasType = Agg;
}

namespace free_functions {
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_15(free_functions::tests::NamespaceTypedefType start, free_functions::tests::NamespaceTypedefType *ptr) {
}
}

namespace free_functions {
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_16(free_functions::tests::AliasType start, free_functions::tests::AliasType *ptr) {
}
}

namespace free_functions {
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_17(AliasType start, AliasType *ptr) {
}
}

namespace free_functions {
  struct Agg {
    int a;
    float b;
  };
}

namespace free_functions::tests {
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_18(free_functions::Agg start, free_functions::Agg *ptr) {
  ptr->a = start.a + 1;
  ptr->b = start.b + 1.1f;
}
}

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_19(free_functions::AliasStruct KArg) {
  for (int j = 0; j < TestArrSizeAlias; j++)
    for (int i = KArg.start[j]; i <= KArg.end[j]; i++)
      KArg.data[j][i] = KArg.start[j] + KArg.end[j];
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_20(sycl::accessor<int, 1, sycl::access::mode::read_write> acc) {
}

// CHECK:      const char* const kernel_names[] = {
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_2Piii
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_2Piiii
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_3IiEvPT_S0_S0_
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_3IfEvPT_S0_S0_
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_3IdEvPT_S0_S0_
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_410NoPointers8Pointers3Agg
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_6I3Agg7DerivedEvT_T0_i
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_7ILi3EEv16KArgWithPtrArrayIXT_EE
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_8N4sycl3_V117work_group_memoryIiEE

// CHECK-NEXT:   {{.*}}__sycl_kernel_free_functions4ff_9EiPi
// CHECK-NEXT:   {{.*}}__sycl_kernel_free_functions5tests5ff_10EiPi
// CHECK-NEXT:   {{.*}}__sycl_kernel_free_functions5tests2V15ff_11EiPi
// CHECK-NEXT:   {{.*}}__sycl_kernel__GLOBAL__N_15ff_12EiPi
// CHECK-NEXT:   {{.*}}__sycl_kernel_free_functions5ff_13EiPi
// CHECK-NEXT:   {{.*}}__sycl_kernel_free_functions5tests5ff_13EiPi

// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_9N4sycl3_V125dynamic_work_group_memoryIiEE
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_11N4sycl3_V114local_accessorIiLi1EEE
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_11IfEvN4sycl3_V114local_accessorIT_Li1EEE
// CHECK-NEXT:   {{.*}}sycl_kernel_ff_12N4sycl3_V17samplerE
// CHECK-NEXT:   {{.*}}sycl_kernel_ff_13N4sycl3_V16streamE
// CHECK-NEXT:   {{.*}}sycl_kernel_ff_14N4sycl3_V13ext6oneapi12experimental13annotated_argIiJEEE
// CHECK-NEXT:   {{.*}}sycl_kernel_ff_15N4sycl3_V13ext6oneapi12experimental13annotated_ptrIiJEEE
// CHECK-NEXT:   {{.*}}__sycl_kernel_free_functions5tests5ff_14EiPi
// CHECK-NEXT:   {{.*}}__sycl_kernel_free_functions5ff_15EiPi
// CHECK-NEXT:   {{.*}}__sycl_kernel_free_functions5ff_16E3AggPS0_
// CHECK-NEXT:   {{.*}}__sycl_kernel_free_functions5ff_17E7DerivedPS0_
// CHECK-NEXT:   {{.*}}__sycl_kernel_free_functions5tests5ff_18ENS_3AggEPS1_
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_19N14free_functions16KArgWithPtrArrayILi50EEE
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_20N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE

// CHECK-NEXT:   ""
// CHECK-NEXT: };

// CHECK:      const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_2Piii
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12 },

// CHECK:        {{.*}}__sycl_kernel_ff_2Piiii
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 16 },

// CHECK:        {{.*}}__sycl_kernel_ff_3IiEvPT_S0_S0_
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12 },

// CHECK:        {{.*}}__sycl_kernel_ff_3IfEvPT_S0_S0_
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12 },

// CHECK:        {{.*}}__sycl_kernel_ff_3IdEvPT_S0_S0_
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 16 },

// CHECK:  //--- _Z18__sycl_kernel_ff_410NoPointers8Pointers3Agg
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 16, 4 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 32, 20 },

// CHECK:  //--- _Z18__sycl_kernel_ff_6I3Agg7DerivedEvT_T0_i
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 32, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 40, 32 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 72 },

// CHECK:  //--- _Z18__sycl_kernel_ff_7ILi3EEv16KArgWithPtrArrayIXT_EE
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 48, 0 },

// CHECK:  //--- _Z18__sycl_kernel_ff_8N4sycl3_V117work_group_memoryIiEE
// CHECK-NEXT:  { kernel_param_kind_t::kind_work_group_memory, 8, 0 },


// CHECK:  //--- _ZN28__sycl_kernel_free_functions4ff_9EiPi
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 4 },

// CHECK:  //--- _ZN28__sycl_kernel_free_functions5tests5ff_10EiPi
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 4 },

// CHECK:  //--- _ZN28__sycl_kernel_free_functions5tests2V15ff_11EiPi
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 4 },

// CHECK:  //--- _ZN26__sycl_kernel__GLOBAL__N_15ff_12EiPi
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 4 },

// CHECK:  //--- _ZN28__sycl_kernel_free_functions5ff_13EiPi
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 4 },

// CHECK:  //--- _ZN28__sycl_kernel_free_functions5tests5ff_13EiPi
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 0 }, 
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 4 },

// CHECK:  //--- _Z18__sycl_kernel_ff_9N4sycl3_V125dynamic_work_group_memoryIiEE
// CHECK-NEXT:  { kernel_param_kind_t::kind_dynamic_work_group_memory, 8, 0 },

// CHECK: //--- _Z19__sycl_kernel_ff_11N4sycl3_V114local_accessorIiLi1EEE
// CHECK-NEXT:  { kernel_param_kind_t::kind_accessor, 4064, 0 },

// CHECK: //--- _Z19__sycl_kernel_ff_11IfEvN4sycl3_V114local_accessorIT_Li1EEE
// CHECK-NEXT:  { kernel_param_kind_t::kind_accessor, 4064, 0 },

// CHECK: //--- _Z19__sycl_kernel_ff_12N4sycl3_V17samplerE
// CHECK-NEXT: { kernel_param_kind_t::kind_sampler, 8, 0 },

// CHECK: //--- _Z19__sycl_kernel_ff_13N4sycl3_V16streamE
// CHECK-NEXT: { kernel_param_kind_t::kind_stream, 16, 0 },

// CHECK: //--- _Z19__sycl_kernel_ff_14N4sycl3_V13ext6oneapi12experimental13annotated_argIiJEEE
// CHECK-NEXT: { kernel_param_kind_t::kind_std_layout, 4, 0 },

// CHECK: //--- _Z19__sycl_kernel_ff_15N4sycl3_V13ext6oneapi12experimental13annotated_ptrIiJEEE
// CHECK-NEXT: { kernel_param_kind_t::kind_pointer, 8, 0 },

// CHECK:  //--- _ZN28__sycl_kernel_free_functions5tests5ff_14EiPi
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 4 },

// CHECK:  //--- _ZN28__sycl_kernel_free_functions5ff_15EiPi
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 4, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 4 },

// CHECK:  //--- _ZN28__sycl_kernel_free_functions5ff_16E3AggPS0_
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 32, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 32 },

// CHECK:  //--- _ZN28__sycl_kernel_free_functions5ff_17E7DerivedPS0_
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 40, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 40 },

// CHECK:  //--- _ZN28__sycl_kernel_free_functions5tests5ff_18ENS_3AggEPS1_
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 8, 0 },
// CHECK-NEXT:  { kernel_param_kind_t::kind_pointer, 8, 8 },

// CHECK: //--- _Z19__sycl_kernel_ff_19N14free_functions16KArgWithPtrArrayILi50EEE
// CHECK-NEXT:  { kernel_param_kind_t::kind_std_layout, 800, 0 },

// CHECK: //--- _Z19__sycl_kernel_ff_20N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE
// CHECK-NEXT:  { kernel_param_kind_t::kind_accessor, 4062, 0 },

// CHECK:        { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
// CHECK-NEXT: };

// CHECK: Definition of _Z18__sycl_kernel_ff_2Piii as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:
// CHECK: void ff_2(int * ptr, int start, int end);
// CHECK-NEXT: static constexpr auto __sycl_shim1() {
// CHECK-NEXT:   return (void (*)(int *, int, int))ff_2;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_kernel<__sycl_shim1()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim1()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT: }

// CHECK: Definition of _Z18__sycl_kernel_ff_2Piiii as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:
// CHECK: void ff_2(int * ptr, int start, int end, int value);
// CHECK-NEXT: static constexpr auto __sycl_shim2() {
// CHECK-NEXT:   return (void (*)(int *, int, int, int))ff_2;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_kernel<__sycl_shim2()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim2()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT: }

// CHECK: Definition of _Z18__sycl_kernel_ff_3IiEvPT_S0_S0_ as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:
// CHECK: template <typename T> void ff_3(T * ptr, T start, T end);
// CHECK-NEXT: static constexpr auto __sycl_shim3() {
// CHECK-NEXT:   return (void (*)(int *, int, int))ff_3<int>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_kernel<__sycl_shim3()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim3(), 2> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT: }

// CHECK: Definition of _Z18__sycl_kernel_ff_3IfEvPT_S0_S0_ as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:
// CHECK: template <typename T> void ff_3(T * ptr, T start, T end);
// CHECK-NEXT: static constexpr auto __sycl_shim4() {
// CHECK-NEXT:   return (void (*)(float *, float, float))ff_3<float>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_kernel<__sycl_shim4()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim4(), 2> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT: }

// CHECK: Definition of _Z18__sycl_kernel_ff_3IdEvPT_S0_S0_ as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:
// CHECK: template <typename T> void ff_3(T * ptr, T start, T end);
// CHECK-NEXT: static constexpr auto __sycl_shim5() {
// CHECK-NEXT:   return (void (*)(double *, double, double))ff_3<double>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_kernel<__sycl_shim5()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim5(), 2> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT: }

// CHECK: Definition of _Z18__sycl_kernel_ff_410NoPointers8Pointers3Agg as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:
// CHECK-NEXT: struct NoPointers;
// CHECK-NEXT: struct Pointers;
// CHECK-NEXT: struct Agg;
// CHECK: void ff_4(NoPointers S1, Pointers S2, Agg S3);
// CHECK-NEXT: static constexpr auto __sycl_shim6() {
// CHECK-NEXT:   return (void (*)(struct NoPointers, struct Pointers, struct Agg))ff_4;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim6()> {
// CHECK-NEXT:   static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim6()> {
// CHECK-NEXT:   static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: Definition of _Z18__sycl_kernel_ff_6I3Agg7DerivedEvT_T0_i as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:
// CHECK: struct Derived;
// CHECK: template <typename T1, typename T2> void ff_6(T1 S1, T2 S2, int end);
// CHECK-NEXT: static constexpr auto __sycl_shim7() {
// CHECK-NEXT:   return (void (*)(struct Agg, struct Derived, int))ff_6<struct Agg, struct Derived>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim7()> {
// CHECK-NEXT:   static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim7()> {
// CHECK-NEXT:   static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }
//
// CHECK: Definition of _Z18__sycl_kernel_ff_7ILi3EEv16KArgWithPtrArrayIXT_EE as a free function kernel

// CHECK: Forward declarations of kernel and its argument types:
// CHECK: template <int ArrSize> struct KArgWithPtrArray;
//
// CHECK: template <int ArrSize> void ff_7(KArgWithPtrArray<ArrSize> KArg);
// CHECK-NEXT: static constexpr auto __sycl_shim8() {
// CHECK-NEXT:   return (void (*)(struct KArgWithPtrArray<3>))ff_7<3>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim8()> {
// CHECK-NEXT:   static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim8()> {
// CHECK-NEXT:   static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: Definition of _Z18__sycl_kernel_ff_8N4sycl3_V117work_group_memoryIiEE as a free function kernel

// CHECK: Forward declarations of kernel and its argument types:
// CHECK: template <typename DataT> class work_group_memory;

// CHECK: void ff_8(sycl::work_group_memory<int> );
// CHECK-NEXT: static constexpr auto __sycl_shim9() {
// CHECK-NEXT: return (void (*)(class sycl::work_group_memory<int>))ff_8;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim9()> {
// CHECK-NEXT: static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim9()> {
// CHECK-NEXT: static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }


// CHECK: Definition of _ZN28__sycl_kernel_free_functions4ff_9EiPi as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:
// CHECK: namespace free_functions {
// CHECK-NEXT: void ff_9(int start, int * ptr);
// CHECK-NEXT: } // namespace free_functions

// CHECK: static constexpr auto __sycl_shim10() {
// CHECK-NEXT:   return (void (*)(int, int *))free_functions::ff_9;
// CHECK-NEXT: }

// CHECK: Definition of _ZN28__sycl_kernel_free_functions5tests5ff_10EiPi as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:

// CHECK: namespace free_functions {
// CHECK-NEXT: namespace tests {
// CHECK-NEXT: void ff_10(int start, int * ptr);
// CHECK-NEXT: } // namespace tests
// CHECK-NEXT: } // namespace free_functions
// CHECK: static constexpr auto __sycl_shim11() {
// CHECK-NEXT: return (void (*)(int, int *))free_functions::tests::ff_10;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim11()> {
// CHECK-NEXT: static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim11()> {
// CHECK-NEXT: static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: Definition of _ZN28__sycl_kernel_free_functions5tests2V15ff_11EiPi as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:

// CHECK: namespace free_functions {
// CHECK-NEXT:  namespace tests {
// CHECK-NEXT:  inline namespace V1 {
// CHECK-NEXT:  void ff_11(int start, int * ptr);
// CHECK-NEXT:  } // inline namespace V1
// CHECK-NEXT:  } // namespace tests
// CHECK-NEXT:  } // namespace free_functions
// CHECK:  static constexpr auto __sycl_shim12() {
// CHECK-NEXT:    return (void (*)(int, int *))free_functions::tests::V1::ff_11;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim12()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim12()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: Definition of _ZN26__sycl_kernel__GLOBAL__N_15ff_12EiPi as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:

// CHECK: namespace {
// CHECK-NEXT:  void ff_12(int start, int * ptr);
// CHECK-NEXT:  } // namespace 
// CHECK:  static constexpr auto __sycl_shim13() {
// CHECK-NEXT:    return (void (*)(int, int *))ff_12;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim13()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim13()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: Definition of _ZN28__sycl_kernel_free_functions5ff_13EiPi as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:

// CHECK: namespace free_functions {
// CHECK-NEXT:  void ff_13(int start, int * ptr);
// CHECK-NEXT:  } // namespace free_functions
// CHECK:  static constexpr auto __sycl_shim14() {
// CHECK-NEXT:    return (void (*)(int, int *))free_functions::ff_13;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim14()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim14()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }
 
// CHECK: Definition of _ZN28__sycl_kernel_free_functions5tests5ff_13EiPi as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:

// CHECK: namespace free_functions {
// CHECK-NEXT:  namespace tests {
// CHECK-NEXT:  void ff_13(int start, int * ptr);
// CHECK-NEXT:  } // namespace tests
// CHECK-NEXT:  } // namespace free_functions
// CHECK:  static constexpr auto __sycl_shim15() {
// CHECK-NEXT:    return (void (*)(int, int *))free_functions::tests::ff_13;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim15()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim15()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }


// CHECK:  // Definition of _Z18__sycl_kernel_ff_9N4sycl3_V125dynamic_work_group_memoryIiEE as a free function kernel 
// CHECK: Forward declarations of kernel and its argument types:
// CHECK-NEXT: namespace sycl { inline namespace _V1 {
// CHECK-NEXT: template <typename DataT> class dynamic_work_group_memory;
// CHECK-NEXT: }}

// CHECK: void ff_9(sycl::dynamic_work_group_memory<int> );
// CHECK-NEXT: static constexpr auto __sycl_shim16() {
// CHECK-NEXT: return (void (*)(class sycl::dynamic_work_group_memory<int>))ff_9;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {

// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim16()> {
// CHECK-NEXT: static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim16()> {
// CHECK-NEXT: static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: Forward declarations of kernel and its argument types:
// CHECK-NEXT: namespace sycl { inline namespace _V1 {
// CHECK-NEXT: template <typename dataT, int dimensions> class local_accessor;

// CHECK: void ff_11(sycl::local_accessor<int, 1> lacc);
// CHECK-NEXT: static constexpr auto __sycl_shim
// CHECK-NEXT:  return (void (*)(class sycl::local_accessor<int, 1>))ff_11;

// CHECK: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel
// CHECK-NEXT:  static constexpr bool value = true;

// CHECK: template <>
// CHECK-NEXT:struct ext::oneapi::experimental::is_single_task_kernel
// CHECK-NEXT:  static constexpr bool value = true;

// CHECK: Definition of _Z19__sycl_kernel_ff_11IfEvN4sycl3_V114local_accessorIT_Li1EEE as a free function kernel

// CHECK: Forward declarations of kernel and its argument types:

// CHECK: template <typename DataT> void ff_11(sycl::local_accessor<DataT, 1> lacc);
// CHECK-NEXT: static constexpr auto __sycl_shim
// CHECK-NEXT:  return (void (*)(class sycl::local_accessor<float, 1>))ff_11<float>;

// CHECK: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel
// CHECK-NEXT: static constexpr bool value = true;

// CHECK: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_single_task_kernel
// CHECK-NEXT:  static constexpr bool value = true;

// CHECK: Definition of _Z19__sycl_kernel_ff_12N4sycl3_V17samplerE as a free function kernel

// CHECK: Forward declarations of kernel and its argument types:
// CHECK-NEXT:namespace sycl { inline namespace _V1 {
// CHECK-NEXT: class sampler;

// CHECK: void ff_12(sycl::sampler S);
// CHECK-NEXT: static constexpr auto __sycl_shim
// CHECK-NEXT: return (void (*)(class sycl::sampler))ff_12;

// CHECK: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel
// CHECK-NEXT: static constexpr bool value = true;

// CHECK: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_single_task_kernel
// CHECK-NEXT:  static constexpr bool value = true;

// Definition of _Z19__sycl_kernel_ff_13N4sycl3_V16streamE as a free function kernel

// Forward declarations of kernel and its argument types:
// CHECK: namespace sycl { inline namespace _V1 {
// CHECK-NEXT: class stream;

// CHECK: void ff_13(sycl::stream str);
// CHECK-NEXT: static constexpr auto __sycl_shim
// CHECK-NEXT:  return (void (*)(class sycl::stream))ff_13;

// CHECK: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel
// CHECK-NEXT:  static constexpr bool value = true;

// CHECK: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_single_task_kernel
// CHECK-NEXT:  static constexpr bool value = true;

// Definition of _Z19__sycl_kernel_ff_14N4sycl3_V13ext6oneapi12experimental13annotated_argIiJEEE as a free function kernel

// Forward declarations of kernel and its argument types:
// CHECK: namespace sycl { inline namespace _V1 { namespace ext { namespace oneapi { namespace experimental {
// CHECK-NEXT: template <typename T, typename ...Props> class annotated_arg;

// CHECK: void ff_14(sycl::ext::oneapi::experimental::annotated_arg<int> arg);
// CHECK-NEXT: static constexpr auto __sycl_shim
// CHECK-NEXT:  return (void (*)(class sycl::ext::oneapi::experimental::annotated_arg<int>))ff_14;

// CHECK: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel
// CHECK-NEXT: static constexpr bool value = true;

// CHECK: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_single_task_kernel
// CHECK-NEXT:  static constexpr bool value = true;

// Definition of _Z19__sycl_kernel_ff_15N4sycl3_V13ext6oneapi12experimental13annotated_ptrIiJEEE as a free function kernel

// Forward declarations of kernel and its argument types:
// CHECK: namespace sycl { inline namespace _V1 { namespace ext { namespace oneapi { namespace experimental {
// CHECK-NEXT: template <typename T, typename ...Props> class annotated_ptr;

// CHECK: void ff_15(sycl::ext::oneapi::experimental::annotated_ptr<int> ptr);
// CHECK-NEXT: static constexpr auto __sycl_shim
// CHECK-NEXT:  return (void (*)(class sycl::ext::oneapi::experimental::annotated_ptr<int>))ff_15;

// CHECK: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel
// CHECK-NEXT:  static constexpr bool value = true;

// CHECK: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_single_task_kernel
// CHECK-NEXT:   static constexpr bool value = true;

// CHECK: // Definition of _ZN28__sycl_kernel_free_functions5tests5ff_14EiPi as a free function kernel
// CHECK: // Forward declarations of kernel and its argument types:
// CHECK: namespace free_functions {
// CHECK-NEXT:  namespace tests {
// CHECK-NEXT:  void ff_14(int start, int * ptr);
// CHECK-NEXT:  } // namespace tests
// CHECK-NEXT:  } // namespace free_functions
  
// CHECK:  static constexpr auto __sycl_shim23() {
// CHECK-NEXT:    return (void (*)(int, int *))free_functions::tests::ff_14;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim23()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim23()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: // Definition of _ZN28__sycl_kernel_free_functions5ff_15EiPi as a free function kernel
// CHECK: // Forward declarations of kernel and its argument types:
// CHECK: namespace free_functions {
// CHECK-NEXT:  void ff_15(int start, int * ptr);
// CHECK-NEXT:  } // namespace free_functions
// CHECK:  static constexpr auto __sycl_shim24() {
// CHECK-NEXT:    return (void (*)(int, int *))free_functions::ff_15;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim24()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim24()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: // Definition of _ZN28__sycl_kernel_free_functions5ff_16E3AggPS0_ as a free function kernel
// CHECK: // Forward declarations of kernel and its argument types:
// CHECK: namespace free_functions {
// CHECK-NEXT:  void ff_16(Agg start, Agg * ptr);
// CHECK-NEXT:  } // namespace free_functions
// CHECK:  static constexpr auto __sycl_shim25() {
// CHECK-NEXT:    return (void (*)(struct Agg, struct Agg *))free_functions::ff_16;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim25()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim25()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: // Definition of _ZN28__sycl_kernel_free_functions5ff_17E7DerivedPS0_ as a free function kernel
// CHECK: // Forward declarations of kernel and its argument types:
// CHECK: namespace free_functions {
// CHECK-NEXT:  void ff_17(Derived start, Derived * ptr);
// CHECK-NEXT:  } // namespace free_functions
// CHECK:  static constexpr auto __sycl_shim26() {
// CHECK-NEXT:    return (void (*)(struct Derived, struct Derived *))free_functions::ff_17;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim26()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim26()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: // Definition of _ZN28__sycl_kernel_free_functions5tests5ff_18ENS_3AggEPS1_ as a free function kernel
// CHECK: // Forward declarations of kernel and its argument types:
// CHECK-NEXT: namespace free_functions { 
// CHECK-NEXT:   struct Agg;
// CHECK-NEXT:   }
// CHECK:   namespace free_functions {
// CHECK-NEXT:   namespace tests {
// CHECK-NEXT:   void ff_18(free_functions::Agg start, free_functions::Agg * ptr);
// CHECK-NEXT:   } // namespace tests
// CHECK-NEXT:   } // namespace free_functions
// CHECK:   static constexpr auto __sycl_shim27() {
// CHECK-NEXT:     return (void (*)(struct free_functions::Agg, struct free_functions::Agg *))free_functions::tests::ff_18;
// CHECK-NEXT:   }
// CHECK-NEXT:   namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_kernel<__sycl_shim27()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim27()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   }

// CHECK: // Definition of _Z19__sycl_kernel_ff_19N14free_functions16KArgWithPtrArrayILi50EEE as a free function kernel
// CHECK: // Forward declarations of kernel and its argument types:
// CHECK-NEXT: namespace free_functions { 
// CHECK-NEXT:  template <int ArrSize> struct KArgWithPtrArray;
// CHECK-NEXT:  }
  
// CHECK:  void ff_19(free_functions::KArgWithPtrArray<50> KArg);
// CHECK-NEXT:  static constexpr auto __sycl_shim28() {
// CHECK-NEXT:    return (void (*)(struct free_functions::KArgWithPtrArray<50>))ff_19;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim28()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim28()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: Definition of _Z19__sycl_kernel_ff_20N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:
// CHECK: namespace sycl { inline namespace _V1 { namespace access {
// CHECK-NEXT: enum class mode : int;
// CHECK-NEXT: }}}
// CHECK-NEXT: namespace sycl { inline namespace _V1 { namespace access {
// CHECK-NEXT: enum class target : int;
// CHECK-NEXT: }}}
// CHECK-NEXT: namespace sycl { inline namespace _V1 { namespace access {
// CHECK-NEXT: enum class placeholder : int;
// CHECK-NEXT: }}}
// CHECK-NEXT: namespace sycl { inline namespace _V1 { namespace ext { namespace oneapi {
// CHECK-NEXT: template <typename ...properties> class accessor_property_list;
// CHECK-NEXT: }}}}
// CHECK-NEXT: namespace sycl { inline namespace _V1 {
// CHECK-NEXT: template <typename dataT, int dimensions, sycl::access::mode accessmode, sycl::access::target accessTarget, sycl::access::placeholder isPlaceholder, typename propertyListT> class accessor;
// CHECK-NEXT: }}

// CHECK: void ff_20(sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer, sycl::access::placeholder::false_t, sycl::ext::oneapi::accessor_property_list<> > acc);
// CHECK-NEXT: static constexpr auto __sycl_shim29() {
// CHECK-NEXT:  return (void (*)(class sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer, sycl::access::placeholder::false_t, class sycl::ext::oneapi::accessor_property_list<> >))ff_20;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim29()> {
// CHECK-NEXT: static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim29()> {
// CHECK-NEXT: static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: #include <sycl/kernel_bundle.hpp>

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_2Piii
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim1()>() {
// CHECK-NEXT:     return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_2Piii"});
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_2Piiii
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim2()>() {
// CHECK-NEXT:     return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_2Piiii"});
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_3IiEvPT_S0_S0_
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim3()>() {
// CHECK-NEXT:     return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_3IiEvPT_S0_S0_"});
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_3IfEvPT_S0_S0_
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim4()>() {
// CHECK-NEXT:     return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_3IfEvPT_S0_S0_"});
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_3IdEvPT_S0_S0_
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim5()>() {
// CHECK-NEXT:     return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_3IdEvPT_S0_S0_"});
// CHECK-NEXT: }
// CHECK-NEXT: }

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_410NoPointers8Pointers3Agg
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim6()>() {
// CHECK-NEXT:   return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_410NoPointers8Pointers3Agg"});
// CHECK-NEXT: }
// CHECK-NEXT: }

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_6I3Agg7DerivedEvT_T0_i
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim7()>() {
// CHECK-NEXT:   return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_6I3Agg7DerivedEvT_T0_i"});
// CHECK-NEXT: }
// CHECK-NEXT: }

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_7ILi3EEv16KArgWithPtrArrayIXT_EE
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim8()>() {
// CHECK-NEXT:   return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_7ILi3EEv16KArgWithPtrArrayIXT_EE"});
// CHECK-NEXT: }
// CHECK-NEXT: }

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_8N4sycl3_V117work_group_memoryIiEE
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim9()>() {
// CHECK-NEXT: return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_8N4sycl3_V117work_group_memoryIiEE"});
// CHECK-NEXT: }
// CHECK-NEXT: }


// CHECK: Definition of kernel_id of _ZN28__sycl_kernel_free_functions4ff_9EiPi
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim10()>() {
// CHECK-NEXT:    return sycl::detail::get_kernel_id_impl(std::string_view{"_ZN28__sycl_kernel_free_functions4ff_9EiPi"});
// CHECK-NEXT:  }
// CHECK-NEXT:  }

// CHECK: Definition of kernel_id of _ZN28__sycl_kernel_free_functions5tests5ff_10EiPi
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim11()>() {
// CHECK-NEXT:    return sycl::detail::get_kernel_id_impl(std::string_view{"_ZN28__sycl_kernel_free_functions5tests5ff_10EiPi"});
// CHECK-NEXT:  }
// CHECK-NEXT:  } 

// CHECK: Definition of kernel_id of _ZN28__sycl_kernel_free_functions5tests2V15ff_11EiPi
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim12()>() {
// CHECK-NEXT:    return sycl::detail::get_kernel_id_impl(std::string_view{"_ZN28__sycl_kernel_free_functions5tests2V15ff_11EiPi"});
// CHECK-NEXT:  }
// CHECK-NEXT:  }

// CHECK: Definition of kernel_id of _ZN26__sycl_kernel__GLOBAL__N_15ff_12EiPi
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim13()>() {
// CHECK-NEXT:    return sycl::detail::get_kernel_id_impl(std::string_view{"_ZN26__sycl_kernel__GLOBAL__N_15ff_12EiPi"});
// CHECK-NEXT:  }
// CHECK-NEXT:  }

// CHECK: Definition of kernel_id of _ZN28__sycl_kernel_free_functions5ff_13EiPi
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim14()>() {
// CHECK-NEXT:    return sycl::detail::get_kernel_id_impl(std::string_view{"_ZN28__sycl_kernel_free_functions5ff_13EiPi"});
// CHECK-NEXT:  }
// CHECK-NEXT:  }

// CHECK: Definition of kernel_id of _ZN28__sycl_kernel_free_functions5tests5ff_13EiPi
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim15()>() {
// CHECK-NEXT:    return sycl::detail::get_kernel_id_impl(std::string_view{"_ZN28__sycl_kernel_free_functions5tests5ff_13EiPi"});
// CHECK-NEXT:  }
// CHECK-NEXT:  }

//
// CHECK: // Definition of kernel_id of _Z18__sycl_kernel_ff_9N4sycl3_V125dynamic_work_group_memoryIiEE
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim16()>() {
// CHECK-NEXT:   return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_9N4sycl3_V125dynamic_work_group_memoryIiEE"});

// CHECK: Definition of kernel_id of _Z19__sycl_kernel_ff_11N4sycl3_V114local_accessorIiLi1EEE
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim17()>() {
// CHECK-NEXT:  return sycl::detail::get_kernel_id_impl(std::string_view{"_Z19__sycl_kernel_ff_11N4sycl3_V114local_accessorIiLi1EEE"});

// CHECK: Definition of kernel_id of _Z19__sycl_kernel_ff_11IfEvN4sycl3_V114local_accessorIT_Li1EEE
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim18()>() {
// CHECK-NEXT:  return sycl::detail::get_kernel_id_impl(std::string_view{"_Z19__sycl_kernel_ff_11IfEvN4sycl3_V114local_accessorIT_Li1EEE"});

// CHECK: Definition of kernel_id of _Z19__sycl_kernel_ff_12N4sycl3_V17samplerE
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim19()>() {
// CHECK-NEXT:  return sycl::detail::get_kernel_id_impl(std::string_view{"_Z19__sycl_kernel_ff_12N4sycl3_V17samplerE"});

// CHECK: Definition of kernel_id of _Z19__sycl_kernel_ff_13N4sycl3_V16streamE
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim20()>() {
// CHECK-NEXT:  return sycl::detail::get_kernel_id_impl(std::string_view{"_Z19__sycl_kernel_ff_13N4sycl3_V16streamE"});

// CHECK: Definition of kernel_id of _Z19__sycl_kernel_ff_14N4sycl3_V13ext6oneapi12experimental13annotated_argIiJEEE
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim21()>() {
// CHECK-NEXT:  return sycl::detail::get_kernel_id_impl(std::string_view{"_Z19__sycl_kernel_ff_14N4sycl3_V13ext6oneapi12experimental13annotated_argIiJEEE"})

// CHECK: Definition of kernel_id of _Z19__sycl_kernel_ff_15N4sycl3_V13ext6oneapi12experimental13annotated_ptrIiJEEE
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim22()>() {
// CHECK-NEXT:  return sycl::detail::get_kernel_id_impl(std::string_view{"_Z19__sycl_kernel_ff_15N4sycl3_V13ext6oneapi12experimental13annotated_ptrIiJEEE"});

// CHECK-NEXT: }
// CHECK-NEXT: }

// CHECK: // Definition of kernel_id of _ZN28__sycl_kernel_free_functions5tests5ff_14EiPi
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim23()>() {
// CHECK-NEXT:    return sycl::detail::get_kernel_id_impl(std::string_view{"_ZN28__sycl_kernel_free_functions5tests5ff_14EiPi"});
// CHECK-NEXT:  }
// CHECK-NEXT:  }
  
// CHECK:  // Definition of kernel_id of _ZN28__sycl_kernel_free_functions5ff_15EiPi
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim24()>() {
// CHECK-NEXT:    return sycl::detail::get_kernel_id_impl(std::string_view{"_ZN28__sycl_kernel_free_functions5ff_15EiPi"});
// CHECK-NEXT:  }
// CHECK-NEXT:  }
  
// CHECK:  // Definition of kernel_id of _ZN28__sycl_kernel_free_functions5ff_16E3AggPS0_
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim25()>() {
// CHECK-NEXT:    return sycl::detail::get_kernel_id_impl(std::string_view{"_ZN28__sycl_kernel_free_functions5ff_16E3AggPS0_"});
// CHECK-NEXT:  }
// CHECK-NEXT:  }

// CHECK: // Definition of kernel_id of _ZN28__sycl_kernel_free_functions5ff_17E7DerivedPS0_
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim26()>() {
// CHECK-NEXT:    return sycl::detail::get_kernel_id_impl(std::string_view{"_ZN28__sycl_kernel_free_functions5ff_17E7DerivedPS0_"});
// CHECK-NEXT:  }
// CHECK-NEXT:  }

// CHECK: // Definition of kernel_id of _ZN28__sycl_kernel_free_functions5tests5ff_18ENS_3AggEPS1_
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim27()>() {
// CHECK-NEXT:    return sycl::detail::get_kernel_id_impl(std::string_view{"_ZN28__sycl_kernel_free_functions5tests5ff_18ENS_3AggEPS1_"});
// CHECK-NEXT:  }
// CHECK-NEXT:  }

// CHECK: // Definition of kernel_id of _Z19__sycl_kernel_ff_20N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim29()>() {
// CHECK-NEXT:   return sycl::detail::get_kernel_id_impl(std::string_view{"_Z19__sycl_kernel_ff_20N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE"});
// CHECK-NEXT: }
// CHECK-NEXT: }
