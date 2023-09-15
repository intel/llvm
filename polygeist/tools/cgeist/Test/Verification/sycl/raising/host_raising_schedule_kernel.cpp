// RUN: clang++ -O1 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host | FileCheck %s
// RUN: clang++ -O2 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host | FileCheck %s
// RUN: clang++ -O3 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host | FileCheck %s

#include <sycl/sycl.hpp>

std::vector<float> init(std::size_t);

class KernelName;

// CHECK-LABEL: llvm.func internal @_ZNSt17_Function_handlerIFvRN4sycl3_V17handlerEEZ4mainEUlS3_E_E9_M_invokeERKSt9_Any_dataS3_(
// CHECK:         %[[accessor_2:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"[[accessor_class_name_2:class\.sycl::_V1::accessor.*]]", ({{.*}})>
// CHECK:         %[[accessor_1:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"[[accessor_class_name_1:class\.sycl::_V1::accessor.*]]", ({{.*}})>
// CHECK:         %[[lambda:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"{{.*}}", packed (struct<"[[accessor_class_name_1]]", ({{.*}})>, struct<"[[accessor_class_name_2]]", ({{.*}})>, i16, {{.*}}>)>
// CHECK:         %[[cgf:.*]] = llvm.load %arg0 {{.*}} : !llvm.ptr -> !llvm.ptr

// COM: check that we correctly identified the accessors.
// CHECK:         sycl.host.set_captured %[[lambda]][0] = %[[accessor_1]] : !llvm.ptr, !llvm.ptr (!sycl_accessor_1_21llvm2Evoid_w_gb)
// CHECK:         sycl.host.set_captured %[[lambda]][1] = %[[accessor_2]] : !llvm.ptr, !llvm.ptr (!sycl_accessor_1_21llvm2Evoid_r_gb)

// COM: can't match the constant here because we don't propagate constants through the command group function object yet;
// COM: just ensure that the value corresponds to the CGF's second capture.
// CHECK:         %[[cgf_capture_2_gep:.*]] = llvm.getelementptr inbounds %[[cgf]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"{{.*}}", (ptr, ptr, ptr)>
// CHECK:         %[[cgf_capture_2:.*]] = llvm.load %[[cgf_capture_2_gep]] {{.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[cgf_capture_2_load:.*]] = llvm.load %[[cgf_capture_2]] {{.*}} : !llvm.ptr -> i16
// CHECK:         sycl.host.set_captured %[[lambda]][2] = %[[cgf_capture_2_load]] : !llvm.ptr, i16

// COM: find the handler and the range
// CHECK:         sycl.host.handler.set_nd_range %[[handler:.*]] -> range %[[range:.*]] : !llvm.ptr, !llvm.ptr

// COM: finally, check that the `schedule_kernel` op has been raised
// CHECK:         sycl.host.schedule_kernel %[[handler]] -> @device_functions::@_ZTS10KernelName[range %[[range]]](%[[accessor_1]]: !sycl_accessor_1_21llvm2Evoid_w_gb, %[[accessor_2]]: !sycl_accessor_1_21llvm2Evoid_r_gb, %[[cgf_capture_2_load]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i16) -> ()

int main() {
  constexpr std::size_t N = 1024;
  const std::vector<float> a = init(N);
  std::vector<float> b(N);
  short foo = 123;
  sycl::queue q;
  {
    sycl::buffer<float> buff_a(a);
    sycl::buffer<float> buff_b(b);
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a(buff_a, cgh, sycl::read_only);
      auto acc_b = buff_b.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<KernelName>(N, [=](sycl::id<1> i) {
        acc_b[i] = acc_a[i] + foo;
      });
    });
  }
}
