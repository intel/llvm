# Implementation design for compile time constant properties

This document describes the implementation design for the DPC++ extension
[SYCL\_EXT\_ONEAPI\_PROPERTIES][1], which adds a general mechanism for
specifying properties which are known at compile time.  This extension is not
itself a feature, but rather a building block that can be incorporated into
other features.

[1]: <../extensions/proposed/SYCL_EXT_ONEAPI_PROPERTIES.asciidoc>

There are a number of situations where we plan to use compile-time constant
properties, but this design document does not attempt to address them all.
Rather, it describes the design for each "category" of use and illustrates each
category with a specific feature.  For example `accessor` is used to illustrate
properties that are applied to a kernel argument, but the same technique could
be used for other variables that are captured as kernel arguments.

In all cases, the goal of this design is to explain how a DPC++ program that
uses properties is consumed by the device compiler and eventually represented
in LLVM IR.  This typically involves some logic in the header files which
results in a C++ annotation that contains the properties.  The device compiler
front-end is responsible for consuming this annotation and producing some
corresponding LLVM IR.  One of the goals of this design is to avoid changes to
the front-end each time we add a new property, so the front-end is not required
to understand each property it consumes.  Instead, it follows a mechanical
process for converting properties listed in the C++ annotation into LLVM IR,
and this mechanical process need not be updated when we add new properties.

Once the information about properties is represented in IR, it is available to
compiler passes.  For example, the `sycl-post-link` tool might use a property
in order to perform one of its transformations.  Some properties are consumed
by the DPC++ compiler, but others are transformed into SPIR-V for use by the
JIT compiler.  This design document also describes how this SPIR-V
transformation is done.


## Properties on a global variable type

One use for compile-time properties is with types that are used exclusively
for declaring global variables.  One such example is the
[SYCL\_EXT\_ONEAPI\_DEVICE\_GLOBAL][2] extension:

[2]: <../extensions/proposed/SYCL_EXT_ONEAPI_DEVICE_GLOBAL.asciidoc>

```
namespace sycl::ext::oneapi {

template <typename T, typename PropertyListT = property_list<>>
class device_global {/*...*/};

} // namespace sycl::ext::oneapi
```

The following code illustrates a `device_global` variable that is declared with
two compile-time properties:

```
using sycl::ext::oneapi;

device_global<int,
  property_list_t<
    device_image_scope::value_t,
    host_access::value_t<host_access::access::read>>>
  dm1;
```

The header file represents these properties with an internal C++ attribute
named `[[__sycl_detail__::add_ir_attributes_global_variable()]]` whose value
is a list that is created through a template parameter pack expansion:

```
namespace sycl::ext::oneapi {

template <typename T, typename PropertyListT = property_list<>>
class device_global {/*...*/};

// Partial specialization to make PropertyListT visible as a parameter pack
// of properties.
template <typename T, typename ...Props>
class
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_global_variable(
    Props::meta_name..., Props::meta_value...
    )]]
#endif
  device_global<T, property_list<Props...>> {/*...*/};

} // namespace sycl::ext::oneapi
```

The `[[__sycl_detail__::add_ir_attributes_global_variable()]]` attribute has an
even number of parameters, assuming that the optional "filter list" parameter
is not specified (see below for a description of this parameter).  The first
half of the parameters are the names of the properties, and the second half of
the parameters are the values for those properties.  Each property has exactly
one value, so the property at parameter position 0 corresponds to the value at
position _N / 2_, etc.  To illustrate using the same example as before, the
result of the parameter pack expansion would look like this:

```
namespace sycl::ext::oneapi {

template </* ... */> class
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_attributes_global_variable(
    "sycl-device-image-scope",  // Name of first property
    "sycl-host-access",         // Name of second property
    nullptr,                    // First property has no parameter
    "read"                      // Value of second property
    )]]
#endif
  device_global</* ... */> {/*...*/};

} // namespace sycl::ext::oneapi
```

The device compiler only uses the
`[[__sycl_detail__::add_ir_attributes_global_variable()]]` attribute when the
decorated type is used to create an [LLVM IR global variable][3] and the global
variable's type is either:

* The type that is decorated by the attribute, or
* An array of the type that is decorated by the attribute.

[3]: <https://llvm.org/docs/LangRef.html#global-variables>

The device compiler front-end silently ignores the attribute when the decorated
type is used in any other way.

When the device compiler front-end creates a global variable from the decorated
type as described above, it also adds one IR attribute to the global variable
for each property using
[`GlobalVariable::addAttribute(StringRef, StringRef)`][4].  If the property
value is not already a string, it converts it to a string as described in
[IR representation as IR attributes][5].

[4]: <https://llvm.org/doxygen/classllvm_1_1GlobalVariable.html#a6cee3c634aa5de8c51e6eaa4e41898bc>
[5]: <#ir-representation-as-ir-attributes>

Note that the front-end does not need to understand any of the properties in
order to do this translation.


## Properties on kernel arguments

Another use of compile-time properties is with types that are used to define
kernel arguments.  For example, the [SYCL\_ONEAPI\_accessor\_properties][6]
extension could be redesigned to use compile-time properties.  Such a redesign
might look like:

[6]: <../extensions/supported/SYCL_EXT_ONEAPI_ACCESSOR_PROPERTIES.asciidoc>

```
namespace sycl {

template <typename dataT,
          int dimensions,
          access::mode accessmode,
          access::target accessTarget,
          access::placeholder isPlaceholder,
          typename PropertyListT = ext::oneapi::property_list<>>
class __attribute__((sycl_special_class)) accessor {/* ... */};

} // namespace sycl
```

Typical usage would look like this (showing a hypothetical property named
`foo`):

```
using sycl;
using sycl::ext::oneapi;

accessor acc(buf, cgh, property_list{no_alias_v, foo_v<32>});
```

In the headers the C++ attribute
`[[__sycl_detail__::add_ir_attributes_kernel_parameter()]]` is used to decorate
parameters of the `__init` member function in the corresponding
`sycl_special_class` decorated class. As before, the initial parameters are the
names of the properties and the subsequent parameters are the property values.

```
namespace sycl {

template <typename dataT,
          int dimensions,
          access::mode accessmode,
          access::target accessTarget,
          access::placeholder isPlaceholder,
          typename PropertyListT = ext::oneapi::property_list<>>
class __attribute__((sycl_special_class)) accessor {/* ... */};

// Partial specialization to make PropertyListT visible as a parameter pack
// of properties.
template <typename dataT,
          int dimensions,
          access::mode accessmode,
          access::target accessTarget,
          access::placeholder isPlaceholder,
          typename ...Props>
class __attribute__((sycl_special_class)) accessor<dataT,
                                                   dimensions,
                                                   accessmode,
                                                   accessTarget,
                                                   isPlaceholder,
                                                   property_list<Props...>> {
  dataT *ptr;

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
        Props::meta_name..., Props::meta_value...
        )]]
      dataT *_ptr) {
    ptr = _ptr;
  }
#endif

};

} // namespace sycl
```

Illustrating this with the previous example:

```
namespace sycl {

template </* ... */>
class __attribute__((sycl_special_class)) accessor</* ... */> {
  dataT *ptr;

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
        "sycl-no-alias",  // Name of first property
        "sycl-foo",       // Name of second property
        nullptr,          // First property has no parameter
        32                // Value of second property
        )]]
      dataT *_ptr) {
    ptr = _ptr;
  }
#endif
};

} // namespace sycl
```

As the name implies, this C++ attribute is only used to decorate parameters of
the `__init` member function of a class type that is as SYCL "special class"
(i.e. a class that is decorated with `__attribute__((sycl_special_class))`).
The device compiler front-end ignores the attribute when it is used in any
other syntactic position.

When the front-end creates a kernel argument from a SYCL "special class", it
copies all parameters of the `__init` member function to the corresponding
kernel function.  If a copied parameter is decorated with
`[[__sycl_detail__::add_ir_attributes_kernel_parameter()]]`, the front-end adds
one LLVM IR attribute to the resulting kernel function parameter for each
property in the list.  For example, this can be done by calling
[`Function::addParamAttrs(unsigned ArgNo, const AttrBuilder &)`][7].  As
before, the IR attributes are added as strings, so the front-end must convert
the property value to a string if it is not already a string.

[7]: <https://llvm.org/doxygen/classllvm_1_1Function.html#a092beb46ecce99e6b39628ee92ccd95a>


## Properties on kernel functions

Compile-time properties can also be used to decorate kernel functions as with
the [SYCL\_EXT\_ONEAPI\_PROPERTIES][8] extension.  There are two ways the
application can specify these properties.  The first is by passing a
`property_list` parameter to the function that submits the kernel:

[8]: <../extensions/proposed/SYCL_EXT_ONEAPI_PROPERTIES.asciidoc>

```
namespace sycl {

class handler {
  template <typename KernelName, typename KernelType, typename PropertyListT>
  void single_task(PropertyListT properties, const KernelType &kernelFunc);
};

// namespace sycl
```

For example:

```
using sycl;
using sycl::ext::oneapi;

void foo(handler &cgh) {
  cgh.single_task(
    property_list{sub_group_size_v<32>, device_has_v<aspect::fp16>},
    [=] {/* ... */});
}
```

The second way an application can specify kernel properties is by adding a
`properties` member variable to a named kernel function object:

```
using sycl;
using sycl::ext::oneapi;

class MyKernel {
 public:
  void operator()() {/* ... */}

  static constexpr auto properties =
    property_list{sub_group_size_v<32>, device_has_v<aspect::fp16>};
};

void foo(handler &cgh) {
  MyKernel k;
  cgh.single_task(k);
}
```

Internally, the header lowers both cases to a wrapper class which defines
`operator()`, and that operator function becomes the "top level" kernel
function that is recognized by the front-end.  The definition of this operator
is decorated with the C++ attribute
`[[__sycl_detail__::add_ir_attributes_function()]]`, and the parameters to this
attribute represent the properties.

```
template<typename KernelType, typename PropertyListT>
class KernelSingleTaskWrapper;

// Partial specialization to make PropertyListT visible as a parameter pack
// of properties.
template<typename KernelType, typename ...Props>
class KernelSingleTaskWrapper<KernelType, property_list<Props...>> {
  KernelType k;

 public:
  KernelSingleTaskWrapper(KernelType k) : k(k) {}

#ifdef __SYCL_DEVICE_ONLY__
  [[clang::sycl_kernel]]
  [[__sycl_detail__::add_ir_attributes_function(
    Props::meta_name..., Props::meta_value...
    )]]
#endif
  void operator()() const {k();}
};
```

Although the DPC++ headers only use the
`[[__sycl_detail__::add_ir_attributes_function()]]` attribute on the definition
of a kernel function as shown above, the front-end recognizes it for any
function definition.  The front-end adds one LLVM IR function attribute for
each property in the list.  For example, this can be done by calling
[`Function::addFnAttr(StringRef, StringRef)`][9].  As before, the IR attributes
are added as strings, so the front-end must convert the property value to a
string if it is not already a string.

[9]: <https://llvm.org/doxygen/classllvm_1_1Function.html#ae7b919df259dce5480774e656791c079>

**NOTE**: The intention is to replace the existing member functions like
`handler::kernel_single_task()` with wrapper classes like
`KernelSingleTaskWrapper`.  We believe this will not cause problems for the
device compiler front-end because it recognizes kernel functions via the
`[[clang::sycl_kernel]]` attribute, not by the name
`handler::kernel_single_task()`.


## Properties on a non-global variable type

Another use of compile-time properties is with types that are used to define
non-global variables.  An example of this is the proposed `annotated_ptr`
class.

```
namespace sycl::ext::oneapi {

template <typename T, typename PropertyListT = property_list_t<>>
class annotated_ptr {
  T *ptr;
 public:
  annotated_ptr(T *p) : ptr(p) {}
};

} // namespace sycl::ext::oneapi
```

where an example use looks like:

```
using sycl::ext::oneapi;

void foo(int *p) {
  annotated_ptr<int
    property_list_t<
      foo::value_t,
      bar::value_t<32>>>
    aptr(p);
}
```

We again use a C++ attribute to represent the properties in the header.  The
attribute `[[__sycl_detail__::add_ir_annotations_member()]]` decorates one of
the member variables of the class, and the parameters to this attribute
represent the properties.

```
namespace sycl::ext::oneapi {

template <typename T, typename PropertyListT = property_list_t<>>
class annotated_ptr;

// Partial specialization to make PropertyListT visible as a parameter pack
// of properties.
template <typename T, typename ...Props>
class annotated_ptr<T, property_list<Props...>> {
  T *ptr
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_annotations_member(
    Props::meta_name..., Props::meta_value...
    )]]
#endif
  ;
 public:
  annotated_ptr(T *p) : ptr(p) {}
};

} // namespace sycl::ext::oneapi
```

Illustrating this with properties from our previous example:

```
namespace sycl::ext::oneapi {

template <typename T, typename PropertyListT = property_list_t<>>
class annotated_ptr;

// Partial specialization to make PropertyListT visible as a parameter pack
// of properties.
template <typename T, typename ...Props>
class annotated_ptr<T, property_list<Props...>> {
  T *ptr
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_annotations_member(
    "sycl-foo",   // Name of first property
    "sycl-bar",   // Name of second property
    nullptr,      // First property has no parameter
    32            // Value of second property
    )]]
#endif
  ;
 public:
  annotated_ptr(T *p) : ptr(p) {}
};

} // namespace sycl::ext::oneapi
```

When the device compiler generates code to reference the decorated member
variable, it emits a call to the LLVM intrinsic function
[`@llvm.ptr.annotation`][10] that annotates the pointer to that member
variables, similar to the way the existing `[[clang::annotate()]]` attribute
works.  Illustrating this with some simplified LLVM IR that matches the example
code above:

[10]: <https://llvm.org/docs/LangRef.html#llvm-ptr-annotation-intrinsic>

```
@.str = private unnamed_addr constant [16 x i8] c"sycl-properties\00",
   section "llvm.metadata"
@.str.1 = private unnamed_addr constant [9 x i8] c"file.cpp\00",
   section "llvm.metadata"
@.str.2 = private unnamed_addr constant [9 x i8] c"sycl-foo\00", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"sycl-bar\00", align 1

@.args = private unnamed_addr constant { [9 x i8]*, i8*, [9 x i8]*, i32 }
   {
     [9 x i8]* @.str.2,   ; Name of first property "sycl-foo"
     i8* null,            ; Null indicates this property has no value
     [9 x i8]* @.str.3,   ; Name of second property "sycl-bar"
     i32 32               ; Value of second property
   },
   section "llvm.metadata"

define void @foo(i32* %ptr) {
  %aptr = alloca %class.annotated_ptr
  %ptr = getelementptr inbounds %class.annotated_ptr, %class.annotated_ptr* %aptr,
    i32 0, i32 0
  %1 = bitcast i32** %ptr to i8*

  %2 = call i8* @llvm.ptr.annotation.p0i8(i8* nonnull %0,
    i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str, i64 0, i64 0),
    i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.1, i64 0, i64 0),
    i32 3,
    i8* bitcast ({ [9 x i8]*, i8*, [9 x i8]*, i32 }* @.args to i8*))

  %3 = bitcast i8* %2 to i32**
  store i32* %ptr, i32** %3
  ret void
}
```

The front-end encodes the properties from the C++ attribute
`[[__sycl_detail__::add_ir_annotations_member()]]` into the
`@llvm.ptr.annotation` call as follows:

* The first parameter to `@llvm.ptr.annotation` is the pointer to annotate (as
  with any call to this intrinsic).
* The second parameter is the literal string `"sycl-properties"`.
* The third parameter is the name of the source file (as with any call to this
  intrinsic).
* The fourth parameter is the line number (as with any call to this intrinsic).
* The fifth parameter is a metadata tuple with information about all of the
  properties.  The first element of the tuple is a string literal with the name
  of the first property.  The second element is the value of the first
  property.  The third element is a string literal with the name of the second
  property, etc.  Since each property has exactly one value, this tuple has an
  even number of elements.

**NOTE**: Calls to the `@llvm.ptr.annotation` intrinsic function are known to
disable many clang optimizations.  As a result, properties added to a
non-global variable will likely result in LLVM IR (and SPIR-V) that is not well
optimized.  This puts more pressure on the SPIR-V consumer (e.g. JIT compiler)
to perform these optimizations.


## Property representation in C++ attributes and in IR

As noted above, there are several C++ attributes that convey property names and
values to the front-end:

* `[[__sycl_detail__::add_ir_attributes_global_variable()]]`
* `[[__sycl_detail__::add_ir_attributes_kernel_parameter()]]`
* `[[__sycl_detail__::add_ir_attributes_function()]]`
* `[[__sycl_detail__::add_ir_annotations_member()]]`

All of these attributes take a parameter list with the same format.  There are
always an even number of parameters, where the first half are the property
names and the second half are the property values.  (This assumes that the
initial optional parameter is not passed.  See below for a description of this
optional parameter.) The property name is always a string literal or a
`constexpr char *` expression.  By convention, property names normally start
with the prefix `"sycl-"` in order to avoid collision with non-SYCL IR
attributes, but this is not a strict requirement.

The property value can be a literal or `constexpr` expression of the following
types:

* `const char *`.
* An integer type.
* A floating point type.
* A boolean type.
* A character type.
* An enumeration type.
* `nullptr_t` (reserved for the case when a property has no value).

All properties require a value when represented in the C++ attribute.  If the
SYCL property has no value the header passes `nullptr`.

### IR representation as IR attributes

Properties that are implemented using the following C++ attributes are
represented in LLVM IR as IR attributes:

* `[[__sycl_detail__::add_ir_attributes_global_variable()]]`
* `[[__sycl_detail__::add_ir_attributes_kernel_parameter()]]`
* `[[__sycl_detail__::add_ir_attributes_function()]]`

When the front-end consumes these C++ attributes and produces IR, each property
name becomes an IR attribute name and each property value becomes the
attribute's value.  Because the attribute values must be strings, the front-end
converts each property value to a string.  Integer and floating point values
are converted with the same format as `std::to_string()` would produce.
Boolean values are converted to either `"true"` or `"false"`.  Enumeration
values are first converted to an integer and then converted to a string with
the same format as `std::to_string()`.  The `nullptr` value is converted to an
empty string (`""`).

**TODO**: Should we allow property values that are type names?  If so, I
suppose they would be converted to a string representation of the mangled name?

**TODO**: Should we allow property values of other (non-fundamental) types?  If
we allow this, we need to teach the front-end how to convert each type to a
string, which means the front-end needs to be changed each time we add a
property with a new non-fundamental type.  This seems undesirable.  However, if
we do not allow non-fundamental types, how do we represent properties like
`work_group_size`, whose value is a 3-tuple of integers?  Maybe we could just
allow `std::tuple`, where the type of each element is one of the fundamental
types listed above.

### IR representation via `@llvm.ptr.annotation`

Properties that are implemented using
`[[__sycl_detail__::add_ir_annotations_member()]]`, are represented in LLVM IR
as the fifth metadata parameter to the `@llvm.ptr.annotation` intrinsic
function.  This parameter is a tuple of metadata values with the following
sequence:

* Name of the first property
* Value of the first property
* Name of the second property
* Value of the second property
* Etc.

Since metadata types are not limited to strings, there is no need to convert
the property values to strings.


## Filtering properties

It is sometimes necessary to filter out certain properties so that only a
subset of the properties in a list are represented in IR.  There are two
scenarios when this is useful.

In some cases, a property is used only in the header file itself, and there is
no need to represent that property in LLVM IR.  In order to avoid cluttering
the IR with unneeded information, these properties can be "filtered out", so
that the front-end does not generate an IR representation.

Another case is when a class wants to represent some properties one way in the
IR while representing other properties in another way.  For example, a future
version of `accessor` might pass some properties to
`[[__sycl_detail__::add_ir_attributes_kernel_parameter()]]` while passing other
properties to `[[__sycl_detail__::add_ir_annotations_member()]]`.  Again, the
header wants some way to "filter" the properties, such that some properties are
interpreted as "kernel parameter attributes" while other are interpreted as
"member annotations".

To handle these cases, each of the following C++ attributes takes an optional
first parameter that is a brace-enclosed list of property names:

* `[[__sycl_detail__::add_ir_attributes_global_variable()]]`
* `[[__sycl_detail__::add_ir_attributes_kernel_parameter()]]`
* `[[__sycl_detail__::add_ir_attributes_function()]]`
* `[[__sycl_detail__::add_ir_annotations_member()]]`

Since this brace-enclosed list acts somewhat like an initializer list, the
header must include `<initializer_list>` prior to passing this optional first
parameter.

The front-end treats this list as a "pass list", ignoring any property whose
name is not in the list.  To illustrate, consider the following example where
`accessor` treats some properties as "kernel parameter attributes" and others
as "member annotations":

```
template <typename dataT,
          int dimensions,
          access::mode accessmode,
          access::target accessTarget,
          access::placeholder isPlaceholder,
          typename ...Props>
class __attribute__((sycl_special_class)) accessor<dataT,
                                                   dimensions,
                                                   accessmode,
                                                   accessTarget,
                                                   isPlaceholder,
                                                   property_list<Props...>> {
    T *ptr
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_annotations_member(

      // The properties in this list are "member annotations".
      {"sycl-bar"},

      Props::meta_name..., Props::meta_value...
      )]]
#endif
    ;

#ifdef __SYCL_DEVICE_ONLY__
  void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(

        // The properties in this list are "kernel parameter attributes".
        {"sycl-no-alias", "sycl-foo"},

        Props::meta_name..., Props::meta_value...
        )]]
      dataT *_ptr) {
    ptr = _ptr;
  }
#endif
  }
```


## Representing properties in SPIR-V

There is no mechanical process which converts all LLVM IR attributes to
SPIR-V.  This is because we do not need all properties to be expressed in
SPIR-V and because there is no consistent way to represent properties in
SPIR-V.  Therefore, the `sycl-post-link` tool decides on a case-by-case basis
which properties are translated into SPIR-V and which representation to use.

We use the [SPIR-V LLVM Translator][11] to translate from LLVM IR to SPIR-V,
and that tool defines [idiomatic LLVM IR][12] representations that correspond
to various SPIR-V instructions.  Therefore, the `sycl-post-link` tool can
translate a property into a specific SPIR-V instruction by generating the
corresponding idiomatic LLVM IR.  The following sections describe some common
cases.

[11]: <https://github.com/KhronosGroup/SPIRV-LLVM-Translator>
[12]: <https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst>

### Property on a kernel function

When a property on a kernel function needs to be represented in SPIR-V, we
generally translate the property into a SPIR-V **OpExecutionMode** instruction.
The SPIR-V LLVM Translator has an existing way to generate this instruction
when the LLVM IR contains the named metadata `!spirv.ExecutionMode` as
illustrated below:

```
!spirv.ExecutionMode = !{!0, !1}      ; Each operand in this metadata
                                      ;   represents one OpExectionMode
                                      ;   instruction that will be generated.
!0 = !{void ()* @bar, i32 42}         ; The first operand identifies a kernel
                                      ;   function.  The second operand is the
                                      ;   integer value of a SPIR-V execution
                                      ;   mode.
!1 = !{void ()* @bar, i32 43, i32 3}  ; Any additional operands in the metadata
                                      ;   correspond to "extra operands" to the
                                      ;   OpExecutionMode instruction.  These
                                      ;   operands must be integer literals.
```

### Property on a kernel parameter

When a property on a kernel parameter needs to be represented in SPIR-V, we
generally translate the property into a SPIR-V **OpDecorate** instruction for
the corresponding **OpFunctionParameter** of the kernel function.  Since the
SPIR-V LLVM Translator does not have an existing way to generate these
decorations, we propose the following mechanism.

An LLVM IR function definition may optionally have a metadata kind of
`!spirv.ParameterDecorations`.  If it does, that metadata node must have one
operand for each of the function's parameters.  Each of those operands is
another metadata node that describes the decorations for that parameter.  To
illustrate:

```
define spir_kernel void @MyKernel(%arg1, %arg2) !spirv.ParameterDecorations !0 {
}

!0 = !{!1, !2}            ; Each operand in this metadata represents the
                          ;   decorations for one kernel parameter.
!1 = !{!3, !4}            ; The first kernel parameter has two decorations.
!2 = !{}                  ; The second kernel parameter has no decorations.
!3 = !{i32 7742}          ; This is the integer value of the first decoration.
!4 = !{i32 7743, i32 10}  ; The first operand is the integer value of the
                          ;   second decoration.  Additional operands are
                          ;   "extra operands" to the decoration.  These
                          ;   operands may be either integer literals or string
                          ;   literals.
```

### Property on a global variable

When a property on a global variable needs to be represented in SPIR-V, we
generally translate the property into a SPIR-V **OpDecorate** instruction for
the corresponding module scope (global) **OpVariable**.  Again, there is no
existing mechanism to do this in the SPIR-V LLVM Translator, so we propose the
following mechanism.

An LLVM IR global variable definition may optionally have a metadata kind of
`!spirv.Decorations`.  If it does, that metadata node has one operand for each
of the global variable's decorations.  To illustrate:

```
@MyVariable = global %MyClass !spirv.Decorations !0
!0 = !{!1, !2}            ; Each operand in this metadata represents one
                          ;   decoration on the variable.
!1 = !{i32 7744}          ; This is the integer value of the first decoration.
!2 = !{i32 7745, i32 20}  ; The first operand is the integer value of the
                          ;   second decoration.  Additional operands are
                          ;   "extra operands" to the decoration.  These
                          ;   operands may be either integer literals or string
                          ;   literals.
```

### Property on a structure member of a non-global variable

As we noted earlier, a property on a structure member variable is represented
in LLVM IR as a call to the intrinsic function `@llvm.ptr.annotation`, where
the annotation string is `"sycl-properties"` and the properties are represented
as metadata in the fifth parameter to `@llvm.ptr.annotation`.  In order to
understand how these SYCL properties are translated into SPIR-V, it's useful to
review how a normal (i.e. non-SYCL) call to `@llvm.ptr.annotation` is
translated.

The existing behavior of the SPIR-V LLVM Translator is to translate this call
into one (or both) of the following:

* An **OpDecorate** instruction that decorates the intermediate pointer value
  that is returned by the intrinsic (i.e. the pointer to the member variable).

* An **OpMemberDecorate** instruction that decorates the member variable
  itself.

In both cases, the decoration is a single **UserSemantic** decoration where the
string literal is the same as the string literal in the LLVM annotation.

When a SYCL structure member property needs to be represented in SPIR-V,
however, we prefer to represent each property as an extended SPIR-V decoration
rather than using a **UserSemantic** decoration.  There is no existing
mechanism in the SPIR-V LLVM Translator to generate extended decorations like
this, so we propose the following new mechanism.

When a member variable property needs to be represented in SPIR-V, the
`sycl-post-link` tool converts the `@llvm.ptr.annotation` intrinsic call into a
call to the SPIR-V intrinsic `__spirv_AddMemberDecoration` which has a metadata
function argument that specifies the decorations as illustrated below:

```
%annotated_ptr = call i8* __spirv_AddMemberDecoration(i8* %ptr, metadata !0)

!0 = !{!1, !2}            ; Each operand in this metadata represents one
                          ;   decoration.
!1 = !{i32 7744}          ; This is the integer value of the first decoration.
!2 = !{i32 7745, i32 20}  ; The first operand is the integer value of the
                          ;   second decoration.  Additional operands are
                          ;   "extra operands" to the decoration.  These
                          ;   operands may be either integer literals or string
                          ;   literals.
```
