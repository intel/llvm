// This test checks that the FE generates global variables corresponding to the
// virtual table in the global address space (addrspace(1)) when
// -fsycl-allow-virtual-functions is passed.

// RUN: %clang_cc1 -triple spir64 -fsycl-allow-virtual-functions -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// CHECK: @_ZTVN10__cxxabiv120__si_class_type_infoE = external addrspace(1) global ptr addrspace(1)
// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE = external addrspace(1) global ptr addrspace(1)
// CHECK: @_ZTI4Base = linkonce_odr constant { ptr addrspace(1), ptr } { ptr addrspace(1) getelementptr inbounds (ptr addrspace(1), ptr addrspace(1) @_ZTVN10__cxxabiv117__class_type_infoE, i64 2)
// CHECK: @_ZTI8Derived1 = linkonce_odr constant { ptr addrspace(1), ptr, ptr } { ptr addrspace(1) getelementptr inbounds (ptr addrspace(1), ptr addrspace(1) @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2)

SYCL_EXTERNAL bool rand();

class Base {
   public:
    virtual void display() {}
};

class Derived1 : public Base {
   public:
    void display() {}
};

SYCL_EXTERNAL void test() {
  Derived1 d1;
  Base *b = nullptr;
  if (rand())
    b = &d1;
  b->display();
}
