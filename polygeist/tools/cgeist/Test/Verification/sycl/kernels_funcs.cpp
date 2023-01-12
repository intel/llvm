// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR: gpu.module @device_functions
//
// CHECK-MLIR-DAG:  gpu.func @_ZTS8kernel_1
// CHECK-MLIR-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>,
// CHECK-MLIR-SAME: [[PASSTHROUGH:passthrough = \["convergent", "mustprogress", "noinline", "norecurse", "nounwind", "optnone", \["frame-pointer", "all"\], \["no-trapping-math", "true"\], \["stack-protector-buffer-size", "8"\], \["sycl-module-id", ".*/polygeist/tools/cgeist/Test/Verification/sycl/kernels_funcs.cpp"\]\]]]} {
// CHECK-MLIR-DAG:  gpu.func @_ZTSZZ6host_2vENKUlRN4sycl3_V17handlerEE_clES2_E8kernel_2
// CHECK-MLIR-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, [[PASSTHROUGH]]} {
// CHECK-MLIR-DAG:  func.func @_ZN12StoreWrapperIiLi1ELN4sycl3_V16access4modeE1026EEC1ENS1_8accessorIiLi1ELS3_1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS1_3ext6oneapi22accessor_property_listIJEEEEENS1_2idILi1EEERKi
// CHECK-MLIR-SAME: attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<linkonce_odr>, {{.*}}
// COM: StoreWrapper constructor:
// CHECK-MLIR-DAG: func.func @_ZN12StoreWrapperIiLi1ELN4sycl3_V16access4modeE1026EEclEv
// CHECK-MLIR-SAME: attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<linkonce_odr>, {{.*}}

// COM: StoreWrapper constructor:
// CHECK-LLVM-DAG:      define weak_odr spir_kernel void @_ZTS8kernel_1({{.*}}) #[[FUNCATTRS1:[0-9]+]]
// CHECK-LLVM-DAG:      define weak_odr spir_kernel void @_ZTSZZ6host_2vENKUlRN4sycl3_V17handlerEE_clES2_E8kernel_2({{.*}}) #[[FUNCATTRS1]]

// CHECK-LLVM-DAG: attributes #[[FUNCATTRS1]] = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="{{.*}}/polygeist/tools/cgeist/Test/Verification/sycl/kernels_funcs.cpp" }

template <typename DataT,
          int Dimensions = 1,
          sycl::access::mode AccessMode = (std::is_const_v<DataT>
                                           ? sycl::access_mode::read
                                           : sycl::access_mode::read_write)>
class StoreWrapper {
public:
  StoreWrapper(sycl::accessor<DataT, Dimensions, AccessMode> acc,
               sycl::id<Dimensions> index,
               const DataT& el)
    : acc{acc}, index{index}, el{el} {}

  void operator()() {
    acc[index] = el;
  }

private:
  sycl::accessor<DataT, Dimensions, AccessMode> acc;
  sycl::id<Dimensions> index;
  DataT el;
};

class kernel_1 {
  sycl::accessor<sycl::cl_int, 1, sycl::access::mode::read_write> A;

public:
  kernel_1(sycl::accessor<sycl::cl_int, 1, sycl::access::mode::read_write> A)
    : A(A) {}

  void operator()(sycl::id<1> id) const {
    StoreWrapper W{A, id, 42};
    W();
  }
};

void host_1() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};

  {
    auto buf = sycl::buffer<int, 1>{nullptr, range};
    q.submit([&](sycl::handler &cgh) {
	       auto A = buf.get_access<sycl::access::mode::read_write>(cgh);
	       auto ker =  kernel_1{A};
	       cgh.parallel_for<kernel_1>(range, ker);
	     });
  }
}

void host_2() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};

  {
    auto buf = sycl::buffer<int, 1>{nullptr, range};
    q.submit([&](sycl::handler &cgh) {
	       auto A = buf.get_access<sycl::access::mode::read_write>(cgh);
	       cgh.parallel_for<class kernel_2>(range, [=](sycl::id<1> id) {
							 A[id] = 42;
						       });
	     });
  }
}

SYCL_EXTERNAL void function_1(sycl::item<2, true> item) {
  auto id = item.get_id(0);
}
