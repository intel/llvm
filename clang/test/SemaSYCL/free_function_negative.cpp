// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -verify=expected -fsycl-int-header=%t.h %s

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

// expected-error@+1 {{attribute 'add_ir_attributes_function' is already applied with different arguments}}
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

// expected-error@+2 {{argument type Derived of kernel free function can not be virtually derived}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg(Derived Value) {
}

// expected-error@+2 {{argument type Derived of kernel free function can not be virtually derived}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg1(int a, Derived Value, float b, Derived Value1) {
}

class Derived1 : public Derived {
};

// expected-error@+2 {{argument type Derived1 of kernel free function can not be virtually derived}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg2(Derived1 Value) {
}

class Base1 {};
class Derived2 : public Base1, public virtual Base {
};

// expected-error@+2 {{argument type Derived2 of kernel free function can not be virtually derived}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg3(Derived2 Value) {
}

struct Derived3 : virtual public Base {
};

// expected-error@+2 {{argument type Derived3 of kernel free function can not be virtually derived}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualInheritArg4(Derived3 Value) {
}

class Class {
    public:
    virtual void virtualMethod() {}
};

// expected-error@+2 {{argument type Class of kernel free function can not have virtual methods}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualMethodArg1(Class Value) {
}

class Class1: public Class {
    public:
    void NewMethod() {}
};

// expected-error@+2 {{argument type Class1 of kernel free function can not have virtual methods}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
void VirtualMethodArg2(Class1 Value) {
}

class Class2: Base1 {
    public:
    virtual void virtualMethod() {}
};

// expected-error@+2 {{argument type Class2 of kernel free function can not have virtual methods}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] 
void VirtualMethodArg2(Class2 Value) {
}

class Class3 {
    public:
    virtual ~Class3() {}
};

// expected-error@+2 {{argument type Class3 of kernel free function can not have virtual methods}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] 
void VirtualMethodArg3(Class3 Value) {
}
