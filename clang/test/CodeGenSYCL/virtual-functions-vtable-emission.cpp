// Virtual functions annotated with the 'indirectly_callable' property may be
// defined in a different translation unit than the one which constructs objects
// of the corresponding polymorphic class. Middle-end analyses of virtual
// functions (SYCLVirtualFunctionsAnalysisPass) work by inspecting the vtable
// initializer and the attributes of the functions it references, so both have to
// be available in every translation unit which references the vtable:
//
// - the vtable definition must be emitted even though the key function is
//   defined elsewhere, and it has to be mergeable (linkonce_odr) so that we
//   don't get multiple definitions once all device code is linked together;
// - declarations of virtual functions must carry the "indirectly-callable"
//   attribute, not just their definitions.
//
// Vtables of polymorphic classes without such virtual functions keep the old
// behaviour of not being defined in device code at all.
//
// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device \
// RUN:     -emit-llvm %s -o - | FileCheck %s

using size_t = __SIZE_TYPE__;
void *operator new(size_t, void *Ptr) { return Ptr; }

class Base {
public:
  [[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "set-a")]]
  virtual void increment(int *Data);
};

// 'Derived' neither overrides nor annotates 'increment', but its vtable still
// refers to 'Base::increment', so it has to be emitted as well. It also declares
// its own non-inline virtual function, i.e. it has a key function of its own
// which is defined in another translation unit.
class Derived : public Base {
public:
  virtual void hostOnly();
};

// An ordinary polymorphic class, unrelated to the virtual functions extension.
class Plain {
public:
  virtual void unrelated();
};

SYCL_EXTERNAL void construct(void *Storage) {
  new (Storage) Base();
  new (Storage) Derived();
  new (Storage) Plain();
}

// The vtables are emitted here despite the key functions being defined in other
// translation units, and they reference the declarations of those functions.
// Slots for virtual functions which are not device functions are nulled out.
// CHECK-DAG: @_ZTV4Base = linkonce_odr {{.*}} constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @_ZN4Base9incrementEPi to ptr addrspace(4))] }
// CHECK-DAG: @_ZTV7Derived = linkonce_odr {{.*}} constant { [4 x ptr addrspace(4)] } { [4 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @_ZN4Base9incrementEPi to ptr addrspace(4)), ptr addrspace(4) null] }

// 'Plain' has nothing to do with the virtual functions extension, so no
// definition of its vtable is emitted - only a declaration, exactly as before.
// CHECK-DAG: @_ZTV5Plain = external unnamed_addr addrspace(1) constant { [3 x ptr addrspace(4)] }, align 8

// CHECK: declare {{.*}}spir_func void @_ZN4Base9incrementEPi({{.*}}#[[#DECL_ATTRS:]]
// CHECK: attributes #[[#DECL_ATTRS]] = {{.*}}"indirectly-callable"="set-a"
