// The test checks that FE generates specific virtual table global variables
// with GLOBAL address space (addrspace(1)) if -fsycl-allow-virtual-functions is
// passed.

// RUN: %clang_cc1 -fsycl-allow-virtual-functions -fsycl-is-device -internal-isystem %S/Inputs -emit-llvm %s -o - | FileCheck %s

// CHECK: @_ZTVN10__cxxabiv120__si_class_type_infoE = external addrspace(1) global ptr addrspace(1)
// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE = external addrspace(1) global ptr addrspace(1)
// CHECK: @_ZTI4Base = linkonce_odr constant { ptr addrspace(1), ptr } { ptr addrspace(1) getelementptr inbounds (ptr addrspace(1), ptr addrspace(1) @_ZTVN10__cxxabiv117__class_type_infoE, i64 2)
// CHECK: @_ZTI8Derived1 = linkonce_odr constant { ptr addrspace(1), ptr, ptr } { ptr addrspace(1) getelementptr inbounds (ptr addrspace(1), ptr addrspace(1) @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2)
// CHECK: @_ZTI8Derived2 = linkonce_odr constant { ptr addrspace(1), ptr, ptr } { ptr addrspace(1) getelementptr inbounds (ptr addrspace(1), ptr addrspace(1) @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2)

#include "sycl.hpp"

#define BASE 1
#define DERIVED1 2
#define DERIVED2 3

SYCL_EXTERNAL int rand();

class Base {
   public:
    [[intel::device_indirectly_callable]] virtual int display() {
       return BASE;
    }
};

class Derived1 : public Base {
   public:
    int display() {
       return DERIVED1;
    }
};

class Derived2 : public Base {
   public:
    int display() {
       return DERIVED2;
    }
};

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() {
      Derived1 d1;
      Derived2 d2;
      Base *b = nullptr;
      if (rand() > 50)
        b = &d1;
      else
        b = &d2;
      b->display();
    });
  });
  q.wait();
  return 0;
}
