// RUN: clang++ -O1 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers | FileCheck %s
// RUN: clang++ -O2 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers | FileCheck %s
// RUN: clang++ -O3 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xclang -opaque-pointers | FileCheck %s

#include <sycl/sycl.hpp>

std::vector<float> init(std::size_t);

class KernelName;

// CHECK-LABEL: llvm.func internal @_ZNSt17_Function_handlerIFvRN4sycl3_V17handlerEEZ4mainEUlS3_E_E9_M_invokeERKSt9_Any_dataS3_(
// CHECK:         %[[accessor_2:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"[[accessor_class_name_2:class\.sycl::_V1::accessor.*]]", ({{.*}})>
// CHECK:         %[[accessor_1:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"[[accessor_class_name_1:class\.sycl::_V1::accessor.*]]", ({{.*}})>
// CHECK:         %[[lambda:.*]] = llvm.alloca %{{.*}} x !llvm.struct<"{{.*}}", packed (struct<"[[accessor_class_name_1]]", ({{.*}})>, struct<"[[accessor_class_name_2]]", ({{.*}})>, i16, array<6 x i8>)>

// COM: check that we correctly identify captured accessors and scalars
// CHECK:         sycl.host.set_captured %[[lambda]][0] = %[[accessor_1]] : !llvm.ptr, !llvm.ptr
// CHECK:         sycl.host.set_captured %[[lambda]][1] = %[[accessor_2]] : !llvm.ptr, !llvm.ptr
// CHECK:         sycl.host.set_captured %[[lambda]][2] = %{{.*}} : !llvm.ptr, i16

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
	       cgh.parallel_for<KernelName>(N, [=](sycl::id<1> i) { acc_b[i] = acc_a[i] + foo; });
	     });
  }
}
