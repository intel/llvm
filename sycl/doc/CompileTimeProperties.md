# Implementation design for compile time constant properties

This document describes the implementation design for the DPC++ extension
[SYCL\_EXT\_ONEAPI\_PROPERTY\_LIST][1], which adds a general mechanism for
specifying properties which are known at compile time.  This extension is not
itself a feature, but rather a building block that can be incorporated into
other features.

[1]: <extensions/PropertyList/SYCL_EXT_ONEAPI_property_list.asciidoc>

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

[2]: <extensions/DeviceGlobal/SYCL_INTEL_device_global.asciidoc>

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
named `[[__sycl_detail__::add_ir_global_variable_attributes()]]` whose value
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
  [[__sycl_detail__::add_ir_global_variable_attributes(
    Props::meta_name..., Props::meta_value...
    )]]
#endif
  device_global<T, property_list<Props...>> {/*...*/};

} // namespace sycl::ext::oneapi
```

The initial entries in the C++ attribute's parameter list are the names of the
properties, and these are followed by the values of the properties.  To
illustrate using the same example as before, the result of the parameter pack
expansion would look like this:

```
namespace sycl::ext::oneapi {

template </* ... */> class
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_global_variable_attributes(
    "sycl-device-image-scope",  // Name of first property
    "sycl-host-access",         // Name of second property
    "",                         // First property has no parameter
    "read"                      // Value of second property
    )]]
#endif
  device_global</* ... */> {/*...*/};

} // namespace sycl::ext::oneapi
```

The device compiler only uses the
`[[__sycl_detail__::add_ir_global_variable_attributes()]]` attribute when the
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
[Property representation in C++ attributes][5].

[4]: <https://llvm.org/doxygen/classllvm_1_1GlobalVariable.html#a6cee3c634aa5de8c51e6eaa4e41898bc>
[5]: <#property-representation-in-C-attributes>

Note that the front-end does not need to understand any of the properties in
order to do this translation.


## Properties on kernel arguments

Another use of compile-time properties is with types that are used to define
kernel arguments.  For example, the [SYCL\_ONEAPI\_accessor\_properties][6]
extension could be redesigned to use compile-time properties.  Such a redesign
might look like:

[6]: <https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/accessor_properties/SYCL_ONEAPI_accessor_properties.asciidoc>

```
namespace sycl {

template <typename dataT,
          int dimensions,
          access::mode accessmode,
          access::target accessTarget,
          access::placeholder isPlaceholder,
          typename PropertyListT = ext::oneapi::property_list<>>
class accessor {/* ... */};

} // namespace sycl
```

Typical usage would look like this (showing a hypothetical property named
`foo`):

```
using sycl;
using sycl::ext::oneapi;

accessor acc(buf, cgh, property_list{no_alias_v, foo_v<32>});
```

As before, the header file represents the properties with an internal C++
attribute, where the initial parameters are the names of the properties and
the subsequent parameters are the property values.

```
namespace sycl {

template <typename dataT,
          int dimensions,
          access::mode accessmode,
          access::target accessTarget,
          access::placeholder isPlaceholder,
          typename PropertyListT = ext::oneapi::property_list<>>
class accessor {/* ... */};

// Partial specialization to make PropertyListT visible as a parameter pack
// of properties.
template <typename dataT,
          int dimensions,
          access::mode accessmode,
          access::target accessTarget,
          access::placeholder isPlaceholder,
          typename ...Props>
class
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_kernel_parameter_attributes(
    Props::meta_name..., Props::meta_value...
    )]]
#endif
  accessor<dataT,
           dimensions,
           accessmode,
           accessTarget,
           isPlaceholder,
           property_list<Props...>> {/*...*/};

} // namespace sycl
```

Illustrating this with the previous example:

```
namespace sycl {

template </* ... */> class
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_kernel_parameter_attributes(
    "sycl-no-alias",  // Name of first property
    "sycl-foo",       // Name of second property
    "",               // First property has no parameter
    32                // Value of second property
    )]]
#endif
  accessor</* ... */> {/* ... */};

} // namespace sycl
```

As the name of the C++ attribute suggests, the device compiler front-end uses
the attribute only when the decorated type is the type of a kernel argument,
and it silently ignores the attribute when the class is used in any other way.

When the device compiler front-end creates a kernel argument in this way, it
adds one LLVM IR attribute to the kernel function's parameter for each property
in the list.  For example, this can be done by calling
[`Function::addParamAttrs(unsigned ArgNo, const AttrBuilder &)`][7].  As
before, the IR attributes are added as strings, so the front-end must convert
the property value to a string if it is not already a string.

[7]: <https://llvm.org/doxygen/classllvm_1_1Function.html#a092beb46ecce99e6b39628ee92ccd95a>

**TODO**: What happens when a "sycl special class" object is captured as a
kernel argument?  The compiler passes each member of the class as a separate
argument.  Should the device compiler duplicate the properties on each such
parameter in this case?  Or, is it the header's responsibility to add the C++
attribute to one of the member variables in this case?  How does the header
decide which member variable to decorate, though?


## Properties on kernel functions

Compile-time properties can also be used to decorate kernel functions as with
the [SYCL\_EXT\_ONEAPI\_KERNEL\_PROPERTIES][8] extension.  There are two ways
the application can specify these properties.  The first is by passing a
`property_list` parameter to the function that submits the kernel:

[8]: <https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/KernelProperties/KernelProperties.asciidoc>

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

Internally, the headers lower both cases to a wrapper class that expresses the
properties as an internal C++ attribute, and the `operator()` of this class
becomes the "top level" kernel function that is recognized by the front-end.

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
  __attribute__((sycl_kernel))
  [[__sycl_detail__::add_ir_function_attributes(
    Props::meta_name..., Props::meta_value...
    )]]
#endif
  void operator()() {k();}
};
```

Although the DPC++ headers only use the
`[[__sycl_detail__::add_ir_function_attributes()]]` attribute on the definition
of a kernel function as shown above, the front-end recognizes it for any
function definition.  The front-end adds one LLVM IR function attribute for
each property in the list.  For example, this can be done by calling
[`Function::addFnAttr(StringRef, StringRef)`][9].  As before, the IR attributes
are added as strings, so the front-end must convert the property value to a
string if it is not already a string.

[9]: <https://llvm.org/doxygen/classllvm_1_1Function.html#ae7b919df259dce5480774e656791c079>

**TODO**: The intention is to replace the existing member functions like
`handler::kernel_single_task()` with wrapper classes like
`KernelSingleTaskWrapper`.  Does this pose any problems?  There are comments in
the headers indicating that the front-end recognizes the function
`handler::kernel_single_task()` by name.


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

We again implement the property list in the header via a C++ attribute, though
this time the attribute decorates a member variable of the class:

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
  [[__sycl_detail__::add_ir_member_annotation(
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
  [[__sycl_detail__::add_ir_member_annotation(
    "foo",    // Name of first property
    "bar",    // Name of second property
    "",       // First property has no parameter
    32        // Value of second property
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
variables, similar to the way the existing clang `__attribute__((annotate()))`
works.  Illustrating this with some simplified LLVM IR that matches the example
code above:

[10]: <https://llvm.org/docs/LangRef.html#llvm-ptr-annotation-intrinsic>

```
@.str = private unnamed_addr constant [27 x i8] c"sycl-properties:foo,bar=32\00",
  section "llvm.metadata"
@.str.1 = private unnamed_addr constant [9 x i8] c"file.cpp\00",
  section "llvm.metadata"

define void @foo(i32* %ptr) {
  %aptr = alloca %class.annotated_ptr
  %ptr = getelementptr inbounds %class.annotated_ptr, %class.annotated_ptr* %aptr,
    i32 0, i32 0
  %1 = bitcast i32** %ptr to i8*
  %2 = call i8* @llvm.ptr.annotation.p0i8(i8* %1,
    i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str, i32 0, i32 0),
    i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.1, i32 0, i32 0),
    i32 3, i8* null)
  %3 = bitcast i8* %2 to i32**
  store i32* %ptr, i32** %3
  ret void
}
```

The front-end encodes the properties from the C++ attribute
`[[__sycl_detail__::add_ir_member_annotation()]]` into the annotation string
(`@.str` in the example above) using the following algorithm:

* The property value is converted to a string as specified in
  [Property representation in C++ attributes][5].
* Construct a property definition string for each property:
  - If the property value is the empty string, the property definition is just
    the name of the property.
  - Otherwise, the property definition string is formed by concatenating the
    property name with the equal sign (`=`) and the property value.
* The annotation string is formed by concatenating all property definition
  strings, separated by a comma (`,`).
* The annotation string is pre-pended with `"sycl-properties:"` and NULL
  terminated.

**NOTE**: Calls to the `@llvm.ptr.annotation` intrinsic function are known to
disable many clang optimizations.  As a result, properties added to a
non-global variable will likely result in LLVM IR (and SPIR-V) that is not well
optimized.  This puts more pressure on the SPIR-V consumer (e.g. JIT compiler)
to perform these optimizations.


## Property representation in C++ attributes

As noted above, there are several C++ attributes that convey property names and
values to the front-end:

* `[[__sycl_detail__::add_ir_global_variable_attributes()]]`
* `[[__sycl_detail__::add_ir_kernel_parameter_attributes()]]`
* `[[__sycl_detail__::add_ir_function_attributes()]]`
* `[[__sycl_detail__::add_ir_member_annotation()]]`

All of these attributes take a parameter list with the same format.  There are
always an even number of parameters, where the first half are the property
names and the second half are the property values.  The property name is always
a string literal or a `constexpr char *` expression.  By convention, property
names that correspond to LLVM IR attributes normally start with the prefix
`"sycl-"` in order to avoid collision with non-SYCL IR attributes, but this is
not a strict requirement.

The property value can be a literal or `constexpr` expression of the following
types:

* `const char *`.
* An integer type.
* A floating point type.
* A boolean type.
* A character type.
* An enumeration type.

All properties require a value when represented in the C++ attribute.  If the
SYCL property has no value the header passes the empty string (`""`).

The front-end converts each value to a string before representing it in LLVM
IR.  Integer and floating point values are converted with the same format as
`std::to_string()` would produce.  Boolean values are converted to either
`"true"` or `"false"`.  Enumeration values are first converted to an integer
and then converted to a string with the same format as `std::to_string()`.

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
the annotation string starts with the prefix `"sycl-properties:"`.  In order to
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
