# Implementation design for sycl_ext_oneapi_virtual_functions

Corresponding language extension specification:
[sycl_ext_oneapi_virtual_functions][1]


## Design

### Changes to the SYCL header files

New compile-time properties `indirectly_callable` and `calls_indirectly` should
be implemented in accordance with the corresponding [design document][2].

**TODO**: `calls_indirectly` requires conversion from C++ typename to a string.
Document how it should be done. `__sycl_builtin_unique_stable_name` should
likely be used.
**TODO**: `calls_indirectly` requires compile-time concatenation of strings.
Document how it should be done.

### Changes to the compiler front-end

Compiler front-end should be updated to respect rules defined by the
[extension specifiction][1], such as:

- virtual member functions annotated with `indirectly_callable` compile-time
  property should be emitted into device code;
- virtual member function *not* annotated with `indirectly_callable`
  compile-time property should *not* be emitted into device code;

### Changes to the compiler middle-end

Note: some of the changes attributed to this category could technically be
implemented in front-end instead. However, it would be more complicated



[1]: <../extensions/proposed/sycl_ext_oneapi_virtual_functions.asciidoc>
[2]: <CompileTimeProperties.md>

