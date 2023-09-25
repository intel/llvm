// RUN: clang++ -O1 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host | FileCheck %s
// RUN: clang++ -O2 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host | FileCheck %s
// RUN: clang++ -O3 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host | FileCheck %s

#include <sycl/sycl.hpp>

std::vector<float> init(std::size_t);

class KernelName;

// CHECK-LABEL: llvm.func internal @_ZNSt17_Function_handlerIFvRN4sycl3_V17handlerEEZ4mainEUlS3_E_E9_M_invokeERKSt9_Any_dataS3_(
// CHECK:         %[[bar_const:.*]] = llvm.mlir.constant(456 : i64) : i64
// CHECK:         %[[accessor_2:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"[[accessor_class_name_2:class\.sycl::_V1::accessor.*]]", ({{.*}})>
// CHECK:         %[[accessor_1:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"[[accessor_class_name_1:class\.sycl::_V1::accessor.*]]", ({{.*}})>
// CHECK:         %[[lambda:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"{{.*}}", (struct<"[[accessor_class_name_1]]", ({{.*}})>, struct<"[[accessor_class_name_2]]", ({{.*}})>, i16, i64, struct<"class.sycl::_V1::vec{{.*}}", (struct<"struct.std::array", (array<4 x f32>)>)>, struct<"class.sycl::_V1::vec{{.*}}", ({{.*}})>)>
// CHECK:         %[[cgf:.*]] = llvm.load %arg0 {{.*}} : !llvm.ptr -> !llvm.ptr

// COM: check that we correctly identified the accessors.
// CHECK:         sycl.host.set_captured %[[lambda]][0] = %[[accessor_1]] : !llvm.ptr, !llvm.ptr (!sycl_accessor_1_21llvm2Evoid_w_dev)
// CHECK:         sycl.host.set_captured %[[lambda]][1] = %[[accessor_2]] : !llvm.ptr, !llvm.ptr (!sycl_accessor_1_21llvm2Evoid_r_dev)

// COM: can't match the constant here because we don't propagate constants through the command group function object yet;
// COM: just ensure that the value corresponds to the CGF's second capture.
// CHECK:         %[[cgf_capture_2_gep:.*]] = llvm.getelementptr inbounds %[[cgf]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"{{.*}}", (ptr, ptr, ptr, ptr, ptr)>
// CHECK:         %[[cgf_capture_2:.*]] = llvm.load %[[cgf_capture_2_gep]] {{.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[cgf_capture_2_load:.*]] = llvm.load %[[cgf_capture_2]] {{.*}} : !llvm.ptr -> i16
// CHECK:         sycl.host.set_captured %[[lambda]][2] = %[[cgf_capture_2_load]] : !llvm.ptr, i16

// COM: the constant defined in the CGF can be matched directly.
// CHECK:         sycl.host.set_captured %[[lambda]][3] = %[[bar_const]] : !llvm.ptr, i64

// COM: vectors are passed as pointers to a struct; check that we matched the corresponding memcpy.
// CHECK:         %[[lambda_capture_4_gep:.*]] = llvm.getelementptr inbounds %[[lambda]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<{{.*}}>
// CHECK:         %[[cgf_capture_3_gep:.*]] = llvm.getelementptr inbounds %[[cgf]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"{{.*}}", (ptr, ptr, ptr, ptr, ptr)>
// CHECK:         %[[cgf_capture_3:.*]] = llvm.load %[[cgf_capture_3_gep]] {{.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         "llvm.intr.memcpy"(%[[lambda_capture_4_gep]], %[[cgf_capture_3]], {{.*}}) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:         sycl.host.set_captured %[[lambda]][4] = %[[cgf_capture_3]] : !llvm.ptr, !llvm.ptr

// CHECK:         %[[lambda_capture_5_gep:.*]] = llvm.getelementptr inbounds %[[lambda]][0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<{{.*}}>
// CHECK:         %[[cgf_capture_4_gep:.*]] = llvm.getelementptr inbounds %[[cgf]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"{{.*}}", (ptr, ptr, ptr, ptr, ptr)>
// CHECK:         %[[cgf_capture_4:.*]] = llvm.load %[[cgf_capture_4_gep]] {{.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         "llvm.intr.memcpy"(%[[lambda_capture_5_gep]], %[[cgf_capture_4]], {{.*}}) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:         sycl.host.set_captured %[[lambda]][5] = %[[cgf_capture_4]] : !llvm.ptr, !llvm.ptr

int main() {
  constexpr std::size_t N = 1024;
  const std::vector<float> a = init(N);
  std::vector<float> b(N);
  short foo = 123;
  sycl::float4 vec1{1, 2, 3, 4};
  sycl::half8 vec2{1, 2, 3, 4, 5, 6, 7, 8};
  sycl::queue q;
  {
    sycl::buffer<float> buff_a(a);
    sycl::buffer<float> buff_b(b);
    q.submit([&](sycl::handler &cgh) {
      long bar = 456;
      sycl::accessor acc_a(buff_a, cgh, sycl::read_only);
      auto acc_b = buff_b.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<KernelName>(N, [=](sycl::id<1> i) {
        acc_b[i] = acc_a[i] + foo * bar + vec1[1] - vec2[2];
      });
    });
  }
}
