// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -verify=expected %s

#include "sycl.hpp"

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
foo(int start, ...) { // expected-error {{free function kernel cannot be a variadic function}}
}

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo1(int start, ...) { // expected-error {{free function kernel cannot be a variadic function}}
}

// expected-note@+1 {{conflicting attribute is here}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 1)]] void
foo2(int start);

// expected-error@+1 {{attribute '__sycl_detail__::add_ir_attributes_function' is already applied with different arguments}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo2(int start) {
}

// expected-note@+1 {{previous declaration is here}}
void foo3(int start, int *ptr);

// expected-error@+2 {{the first occurrence of SYCL kernel free function should be declared with 'sycl-nd-range-kernel' or 'sycl-single-task-kernel' compile time properties}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo3(int start, int *ptr){}

// expected-note@+1 {{previous declaration is here}}
void foo4(float start, float *ptr);

// expected-error@+2 {{the first occurrence of SYCL kernel free function should be declared with 'sycl-nd-range-kernel' or 'sycl-single-task-kernel' compile time properties}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo4(float start, float *ptr);

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo4(float start, float *ptr);

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo4(float start, float *ptr){}


// expected-error@+2 {{a function with a default argument value cannot be used to define SYCL free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
singleTaskKernelDefaultValues(int Value = 1) {
}

// expected-error@+2 {{a function with a default argument value cannot be used to define SYCL free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void 
ndRangeKernelDefaultValues(int Value = 1) {
}

// expected-error@+2 {{a function with a default argument value cannot be used to define SYCL free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void  
singleTaskKernelDefaultValues(int Ivalue = 1, unsigned int Uvalue = 3) {
}

// expected-error@+2 {{a function with a default argument value cannot be used to define SYCL free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void 
ndRangeKernelDefaultValues(int Ivalue = 1, unsigned int Uvalue = 3) {
}

// expected-error@+2 {{SYCL free function kernel should have return type 'void'}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] int 
singleTaskKernelReturnType(int Value) {
}

// expected-error@+2 {{SYCL free function kernel should have return type 'void'}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] int 
ndRangeKernelReturnType(int Value) {
}

class TestClass {
public:

// expected-error@+2 {{class method cannot be used to define a SYCL kernel free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void 
ndRangeKernelMethod(int Value) {
}

// expected-error@+2 {{class method cannot be used to define a SYCL kernel free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
singleTaskKernelMethod(int Value) {
}

// expected-error@+2 {{static class method cannot be used to define a SYCL kernel free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
static void StaticndRangeKernelMethod(int Value) {
}

// expected-error@+2 {{static class method cannot be used to define a SYCL kernel free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
static void StaticsingleTaskKernelMethod(int Value) {
}

};

struct TestStruct {

// expected-error@+2 {{class method cannot be used to define a SYCL kernel free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void 
ndRangeKernelMethod(int Value) {
}

// expected-error@+2 {{class method cannot be used to define a SYCL kernel free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
singleTaskKernelMethod(int Value) {
}

// expected-error@+2 {{static class method cannot be used to define a SYCL kernel free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
static void StaticndRangeKernelMethod(int Value) {
}

// expected-error@+2 {{static class method cannot be used to define a SYCL kernel free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
static void StaticsingleTaskKernelMethod(int Value) {
}

};

class Base {};
class Derived : virtual public Base {};

// expected-error@+2 {{argument type 'Derived' virtually inherited from base class `Base` is not supported as a SYCL kernel argument}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg(Derived Value) {
}

// expected-error@+2 1 {{argument type 'Derived' virtually inherited from base class `Base` is not supported as a SYCL kernel argument}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg1(int a, Derived Value, float b, Derived Value1) {
}

class Derived1 : public Derived {
};

// expected-error@+2 {{argument type 'Derived' virtually inherited from base class `Base` is not supported as a SYCL kernel argument}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg2(Derived1 Value) {
}

class Base1 {};
class Derived2 : public Base1, public virtual Base {
};

// expected-error@+2 {{argument type 'Derived2' virtually inherited from base class `Base` is not supported as a SYCL kernel argumen}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg3(Derived2 Value) {
}

template <typename T>
class Derived3 : virtual T {
};

// expected-error@+2 {{argument type 'Derived3' virtually inherited from base class `class Base` is not supported as a SYCL kernel argument}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg4(Derived3<Base> Value) {
}

// expected-error@+3 {{argument type 'Derived3' virtually inherited from base class `class Derived2` is not supported as a SYCL kernel argument}}
// expected-error@+2 {{argument type 'Derived2' virtually inherited from base class `Base` is not supported as a SYCL kernel argument}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg5(Derived3<Derived2> Value) {
}

template <typename T>
class Derived4 : T {
};

// expected-error@+2 {{argument type 'Derived' virtually inherited from base class `Base` is not supported as a SYCL kernel argument}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg6(Derived4<Derived> Value) {
}

// expected-error@+2 {{argument type 'Derived2' virtually inherited from base class `Base` is not supported as a SYCL kernel argument}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg7(Derived4<Derived2> Value) {
}


template <typename T>
class Derived5 : T, virtual Base {
};

// expected-error@+2 {{argument type 'Derived5' virtually inherited from base class `Base` is not supported as a SYCL kernel argument}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg7(Derived5<Base1> Value) {
}

// expected-error@+3 {{argument type 'Derived5' virtually inherited from base class `Base` is not supported as a SYCL kernel argument}}
// expected-error@+2 {{argument type 'Derived' virtually inherited from base class `Base` is not supported as a SYCL kernel argument}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg7(Derived5<Derived1> Value) {
}
