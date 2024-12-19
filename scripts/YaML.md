This document describes the YaML format used by the scripts for the API specifications.

## Input Structure
* The scripts search for all `.yml` files in the folders specified in `config.ini`.
* Each yml file represents a collection of APIs to be generated.
  * It is recommended that each yml file represents a single feature definition, but this is not required.
  * However, each extensions definition must be limited to a single yml file.
* The name of the yml file will be used as C++ `#pragma region` name in the specification.

## YML Syntax
* Each document in the yml file represents an entry in the specification.
* Every document must have a `type` scalar: {`header`, `macro`, `typedef`, `const`, `enum`, `struct`, `handle`, `function`, `class`}
* All scalars must be strings. The document writer is responsible for using explicit string notification where the yml parser may perform implicit conversion.
* Custom names must be tagged using `$` followed by the tag name. The tag names are defined in the `config.ini` section. There are two tag variations for replacement:
    - `$x` : lower_case
    - `$X` : ALL_CAPS

#### type: header
* Every yml file is required to have a header
* A header requires the following scalar fields: {`desc`}
  - `desc` will be used as the region description comment
* A header may take the following optional scalar fields: {`ordinal`, `version`}
  - `ordinal` will be used to override the default order (alphabetical) in which regions appear in the specification; `default="1000"`. Multiple regions with the same ordinal will be ordered alphabetically.
  - `version` can be used to define the minimum API version for all documents in the yml file; `default="1.0"`

<table>
<tr><th>YaML</th><th>C\C++</th><th>Python</th></tr>
<tr><td>

```yml
type: header
desc: "A brief description..."
```
</td>
<td>

```cpp
// A brief description...
#pragma region name
```
</td>
<td>
n/a
</td></tr>
</table>

#### type: macro
* A macro initiates the creation of a C/C++ preprocessor `#define` directive in the specification
* A macro requires the following scalar fields: {`desc`, `name`, `value`}
  - `desc` will be used as the macro's description comment
  - `name` must be a unique ISO-C standard identifier, start with `$` tag and be all caps
  - `name` may optionally contain an ISO-C preprocessor parameter list
  - `value` must be an ISO-C standard preprocessor replacement list
* A macro may take the following optional scalar fields: {`condition`, `altvalue`, `ordinal`, `version`}
  - `condition` will be used as a C/C++ preprocessor `#if` conditional expression
  - `altvalue` must be an ISO-C standard preprocessor replacement list. If specified, then will be contained within the `#else` conditional block.
  - `ordinal` will be used to override the default order (in which they appear) the macro appears within its section; `default="1000"`
  - `version` will be used to define the minimum API version in which the macro will appear; `default="1.0"` This will also affect the order in which the macro appears within its section.
* A macro may take the following optional field which can be a scalar, a sequence of scalars or scalars to sequences: {`details`}
  - `details` will be used as the macro's detailed comment

<table>
<tr><th>YaML</th><th>C\C++</th><th>Python</th></tr>
<tr><td>

```yml
type: macro
desc: "A brief description..."
name: "$X_NAME"
value: "0"
```
</td>
<td>

```cpp
/// @brief A brief description...
#define UR_NAME 0
```
</td>
<td>

```python
## @brief A brief description...
UR_NAME = 0
```
</td></tr>
<tr><td>

```yml
type: macro
desc: "A brief description..."
name: "$X_NAME( P0, P1 )"
value: "P0+P1"
```
</td>
<td>

```cpp
/// @brief A brief description...
#define UR_NAME( P0, P1 ) P0+P1
```
</td>
<td>

```python
## @brief A brief description...
def UR_NAME( P0, P1 ):
    return P0+P1
```
</td></tr>
<tr><td>

```yml
type: macro
desc: "A brief description..."
name: "$X_NAME( P0, P1 )"
condition: "defined(_WIN32)"
value: "P0"
altvalue: "P1"
```
</td>
<td>

```cpp
/// @brief A brief description...
#if defined(_WIN32)
#define UR_NAME( P0, P1 ) P0
#else
#define UR_NAME( P0, P1 ) P1
#endif
```
</td>
<td>
n/a
</td></tr>
</table>

#### type: typedef
* A typedef initiates the creation of a C/C++ `typedef` declaration in the specification
* A typedef requires the following scalar fields: {`desc`, `name`, `value`}
  - `desc` will be used as the typedef's description comment
  - `name` must be a unique ISO-C standard identifier, start with `$` tag, be snake_case and end with `_t`
  - `value` must be an ISO-C standard identifier
* A typedef may take the following optional scalar fields: {`class`, `condition`, `altvalue`, `ordinal`, `version`}
  - `class` will be used to scope the typedef declaration within the specified C++ class
  - `condition` will be used as a C/C++ preprocessor `#if` conditional expression
  - `altvalue` must be an ISO-C standard identifier. If specified, then will be contained within the `#else` conditional block.
  - `ordinal` will be used to override the default order (in which they appear) the typedef appears within its section; `default="1000"`
  - `version` will be used to define the minimum API version in which the typedef will appear; `default="1.0"` This will also affect the order in which the typedef appears within its section and class.
* A typedef may take the following optional field which can be a scalar, a sequence of scalars or scalars to sequences: {`details`}
  - `details` will be used as the typedef's detailed comment

<table>
<tr><th>YaML</th><th>C</th><th>C++</th><th>Python</th></tr>
<tr><td>

```yml
type: typedef
desc: "A brief description..."
name: $x_name_t
value: "void*"
class: $xClsName
```
</td>
<td>

```c
/// @brief A brief description...
typedef void* ur_name_t;
```
</td>
<td>

```cpp
namespace ur {
  class ClsName
  {
    /// @brief A brief description...
    using name_t = void*;

```
</td>
<td>

```python
## @brief A brief description...
class ur_name_t(c_void_p):
    pass
```
</td></tr>
</table>

#### type: fptr_typedef
* A fptr_typedef initiates the creation of a C/C++ `typedef` of a function pointer declaration in the specification
* A fptr_typedef requires the following scalar fields: {`desc`, `name`, `return`}
  - `desc` will be used as the typedef's description comment
  - `name` must be a unique ISO-C standard identifier, start with `$` tag, be snake_case and end with `_t`
  - `return` must be an ISO-C standard data type
* A function pointer typedef can optionally accept the following sequence of mappings: {`params`}
  - A param requires the following scalar fields: {`desc`, `type`, `name`}
  - `desc` will be used as param's description comment
  - `name` must be a unique ISO-C standard identifier
  - `type` must be ISO-C standard data type

<table>
<tr><th>YaML</th><th>C</th><th>C++</th></tr>
<tr><td>

```yml
type: fptr_typedef
desc: "A brief description..."
name: $x_callback_t
return: "void"
params:
    - type: void*
      name: pParams
      desc: "Brief description of param"
```
</td>
<td>

```c
/// @brief A brief description...
typedef void(ur_callback_t)(void* pParams);
```
</td>
<td>

```cpp
std::function<void(void*)> ur_callback_t;
```
</td>
</tr>
</table>

#### type: handle
* A handle initiates the creation of a C/C++ forwarding `struct` pointer declaration or C++ `class` declaration in the specification
* A handle requires the following scalar fields: {`desc`, `name`}
  - `desc` will be used as the handles's description comment
  - `name` must be a unique ISO-C standard identifier, start with `$` tag, be snake_case and end with `_handle_t`
* A handle may take the following optional scalar fields: {`class`, `alias`, `condition`, `ordinal`, `version`, `loader_only`}
  - `class` will be used to scope the handles declaration within the specified C++ class
  - `alias` will be used to declare the handle as an alias of another handle; specifically, aliases in another namespace
  - `condition` will be used as a C/C++ preprocessor `#if` conditional expression
  - `ordinal` will be used to override the default order (in which they appear) the handles appears within its section; `default="1000"`
  - `version` will be used to define the minimum API version in which the handles will appear; `default="1.0"` This will also affect the order in which the handles appears within its section.
  - `loader_only` will be used to decide whether the handle can be instantiated and managed only by the loader.
* A handle may take the following optional field which can be a scalar, a sequence of scalars or scalars to sequences: {`details`}
  - `details` will be used as the handle's detailed comment

<table>
<tr><th>YaML</th><th>C</th><th>C++</th><th>Python</th></tr>
<tr><td>

```yml
type: handle
desc: "A brief description..."
name: $x_name_handle_t
```
</td>
<td>

```c
typedef struct _ur_name_handle_t *ur_name_handle_t;
```
</td>
<td>

```cpp
namespace ur {
  struct _name_handle_t;
  using name_handle_t = _name_handle_t*;

```
</td>
<td>

```python
## @brief A brief description...
class ur_name_handle_t(c_void_p):
    pass
```
</td></tr>
<tr><td>

```yml
type: handle
desc: "A brief description..."
name: $x_name_handle_t
class: $xClsName
```
</td>
<td>

```c
typedef struct _ur_name_handle_t *ur_name_handle_t;
```
</td>
<td>

```cpp
namespace ur {
  class ClsName;
```
</td>
<td>

```python
## @brief A brief description...
class ur_name_handle_t(c_void_p):
    pass
```
</td></tr>
</table>

#### type: enum

In the following section the word *enumerator* is abbreviated to `etor` and the
plural form *enumerators* is abbreviated to `etors`.

* An enum initiates the creation of a C/C++ `enum` declaration in the specification
* An enum requires the following scalar fields: {`desc`, `name`}
  - `desc` will be used as the enum's description comment
  - `name` must be a unique ISO-C standard identifier, start with `$` tag, be snake_case and end with `_t`
  - `name` that endswith `_flags_t` will be used to create bitfields
* An enum may take the following optional scalar fields: {`class`, `condition`, `ordinal`, `version`, `typed_etors`, `extend`}
  - `class` will be used to scope the enum declaration within the specified C++ class
  - `condition` will be used as a C/C++ preprocessor `#if` conditional expression
  - `ordinal` will be used to override the default order (in which they appear) the enum appears within its section; `default="1000"`
  - `version` will be used to define the minimum API version in which the enum will appear; `default="1.0"` This will also affect the order in which the enum appears within its section and class.
  - `extend` will be used to extend an existing enum with additional `etors`, 
  usually used to implement experimental features. `type` *must* refer to an 
  existing enum and each `etor` must include a unique `value`.
  - `typed_etors` boolean value that will be used to determine whether the enum's values have associated types.
* An enum requires the following sequence of mappings: {`etors`}
  - An etor requires the following scalar fields: {`name`, `desc`}
    + `desc` will be used as the etors's description comment
    + If the enum has `typed_etors`, `desc` must begin with type identifier: {`"[type]"`}
    + `desc` may contain the [optional-query] annotation. This denotes the etor as an info query which is optional for adapters to implement, and may legally result in a non-success error code.
    + `name` must be a unique ISO-C standard identifier, and be all caps
  - An etor may take the following optional scalar field: {`value`, `version`}
    + `value` must be an ISO-C standard identifier
    + `version` will be used to define the minimum API version in which the etor will appear; `default="1.0"` This will also affect the order in which the etor appears within the enum.
* An enum may take the following optional field which can be a scalar, a sequence of scalars or scalars to sequences: {`details`}
  - `details` will be used as the enum's detailed comment

<table>
<tr><th>YaML</th><th>C</th><th>C++</th><th>Python</th></tr>
<tr><td>

```yml
type: enum
desc: "A brief description..."
name: $x_name_t
class: $xClsName
etors:
    - name: VALUE_0
      value: "0"
      desc: "brief description"
    - name: VALUE_1
      desc: "brief description"
```
</td>
<td>

```c
/// @brief A brief description...
typedef enum _ur_name_t
{
    UR_NAME_VALUE_0 = 0, ///< brief description
    UR_NAME_VALUE_1      ///< brief description
} ur_name_t;
```
</td>
<td>

```cpp
namespace ur {
  class ClsName
  {
    /// @brief A brief description...
    enum class name_t : uint32_t
    {
        VALUE_0 = 0,   ///< brief description
        VALUE_1        ///< brief description
    };
```
</td>
<td>

```python
## @brief A brief description...
class ur_name_v(IntEnum):
    VALUE_0 = 0,        ## brief description
    VALUE_1 = auto()    ## brief description

```
</td></tr>
<tr><td>

```yml
type: enum
desc: "A brief description..."
name: $x_name_flags_t
class: $xClsName
etors:
    - name: VALUE_0
      desc: "brief description"
    - name: VALUE_1
      desc: "brief description"
```
</td>
<td>

```c
/// @brief A brief description...
typedef uint32_t ur_name_flags_t;
typedef enum _ur_name_flag_t
{
    UR_NAME_FLAG_VALUE_0 = UR_BIT(0), ///< brief description
    UR_NAME_FLAG_VALUE_1 = UR_BIT(1)  ///< brief description
} ur_name_flag_t;
```
</td>
<td>

```cpp
namespace ur {
  class ClsName
  {
    /// @brief A brief description...
    struct name_flags_t
    {
        uint32_t value;
    };

    enum name_flag_t
    {
        NAME_FLAG_VALUE_0 = UR_BIT(0), ///< brief description
        NAME_FLAG_VALUE_1 = UR_BIT(1)  ///< brief description
    };
```
</td>
<td>

```python
## @brief A brief description...
class ur_name_flags_v(IntEnum):
    VALUE_0 = 1h,   ## brief description
    VALUE_1 = 2h    ## brief description

```
</td></tr>
</table>

#### type: struct|union
* A struct|union initiates the creation of a C/C++ `struct` or `union` declaration in the specification
* A struct|union requires the following scalar fields: {`desc`, `name`}
  - `desc` will be used as the struct|union's description comment
  - `name` must be a unique ISO-C standard identifier, start with `$` tag, be snake_case and end with `_t`
    + The special-case descriptor struct should always end with `_desc_t`
    + The special-case property struct should always end with `_properties_t`
* A union requires the following 
  - `tag` is a reference to an enum type that will be used to describe which field of the union to access.
* A struct|union may take the following optional scalar fields: {`class`, `base`, `condition`, `ordinal`, `version`}
  - `class` will be used to scope the struct|union declaration within the specified C++ class
  - `base` will be used as the base type of the structure
  - `condition` will be used as a C/C++ preprocessor `#if` conditional expression
  - `ordinal` will be used to override the default order (in which they appear) the struct|union appears within its section; `default="1000"`
  - `version` will be used to define the minimum API version in which the struct|union will appear; `default="1.0"` This will also affect the order in which the struct|union appears within its section and class.
* A struct|union requires the following sequence of mappings: {`members`}
  - A member requires the following scalar fields: {`desc`, `type`, `name`}
    + `desc` will be used as the members's description comment
    + `desc` must begin with one the following annotations: {`"[in]"`, `"[out]"`, `"[in,out]"`, `"[nocheck]"`} 
      - `in` is used for members that are read-only; if the member is a pointer, then the memory being pointed to is also read-only
      - `out` is used for members that are write-only; if the member is a pointer, then the memory being pointed to is also write-only
      - `in,out` is used for members that are both read and write; typically this is used for pointers to other data structures that contain both read and write members
      - `nocheck` is used to specify that no additional validation checks will be generated.
    + `desc` must also include the following annotation when describing a union: {`"tagged_by(param)"`}
      - `tagged_by` is used to specify which parameter will be used as the tag for accessing the union.
    + `desc` may include one the following annotations: {`"[optional]"`, `"[typename(typeVarName, sizeVarName)]"`}
      - `optional` is used for members that are pointers where it is legal for the value to be `nullptr`
      - `typename` is used to denote the type enum for params that are opaque pointers to values of tagged data types.
    + `type` must be an ISO-C standard identifier; except it may **not** be a `handle_t`
    + `name` must be a unique ISO-C standard identifier
  - A member may take the following optional scalar field: {`init`, `version`}
    + `init` will be used to initialize the C++ struct|union member's value
    + `init` must be an ISO-C standard identifier or literal
    + `version` will be used to define the minimum API version in which the member will appear; `default="1.0"` This will also affect the order in which the member appears within the struct|union.
    + `tag` applies only to unions and refers to a value for when this member can be accessed.
* A struct|union may take the following optional field which can be a scalar, a sequence of scalars or scalars to sequences: {`details`}
  - `details` will be used as the struct|union's detailed comment

<table>
<tr><th>YaML</th><th>C</th><th>C++</th><th>Python</th></tr>
<tr><td>

```yml
type: struct
desc: "A brief description..."
name: $x_name_t
class: $xClsName
members:
    - type: uint32_t
      name: val0
      desc: "brief description"
      init: "0"
    - type: float
      name: val1
      desc: "brief description"
```
</td>
<td>

```c
/// @brief A brief description...
typedef struct _ur_name_t
{
   uint32_t val0; ///< brief description
   float    val1; ///< brief description
} ur_name_t;
```
</td>
<td>

```cpp
namespace ur {
  class ClsName
  {
    /// @brief A brief description...
    struct name_t
    {
       uint32_t val0 = 0; ///< brief description
       float    val1;     ///< brief description
    };
```
</td>
<td>

```python
## @brief A brief description...
class ur_name_t(Structure):
    _fields_ = [
        ("val0", c_ulong),  ## brief description
        ("val1", c_float)   ## brief description
    ]
```
</td></tr>
<tr><td>

```yml
type: union
desc: "A brief description..."
name: $x_name_t
class: $xClsName
members:
    - type: uint32_t
      name: val0
      desc: "brief description"
    - type: float
      name: val1
      desc: "brief description"
```
</td>
<td>

```c
/// @brief A brief description...
typedef union _ur_name_t
{
   uint32_t val0; ///< brief description
   float    val1; ///< brief description
} ur_name_t;
```
</td>
<td>

```cpp
namespace ur {
  class ClsName
  {
    /// @brief A brief description...
    union name_t
    {
       uint32_t val0 = 0; ///< brief description
       float    val1;     ///< brief description
    };
```
</td>
<td>

```python
## @brief A brief description...
class ur_name_t(Structure):
    _fields_ = [
        ("val0", c_ulong),  ## brief description
        ("val1", c_float)   ## brief description
    ]
```
</td></tr>
</table>

#### type: function
* A function initiates the creation of a C/C++ function declaration in the specification
* A function requires the following scalar fields: {`desc`, `name`}
  - `desc` will be used as the function's description comment
  - `name` must be a unique ISO-C standard identifier, and be PascalCase
* A function may take the following optional scalar fields: {`class`, `decl`, `condition`, `ordinal`, `version`, `loader_only`}
  - `class` will be used to scope the function declaration within the specified C++ class
  - `decl` will be used to specify the function's linkage as one of the following: {`static`}
  - `condition` will be used as a C/C++ preprocessor `#if` conditional expression
  - `ordinal` will be used to override the default order (in which they appear) the function appears within its section; `default="1000"`
  - `version` will be used to define the minimum API version in which the function will appear; `default="1.0"` This will also affect the order in which the function appears within its section and class.
  - `loader_only` will be used to decide whether the function will only be implemented by the loader and not appear in the adapters
  interface.
* A function requires the following sequence of mappings: {`params`}
  - A param requires the following scalar fields: {`desc`, `type`, `name`}
    + `desc` will be used as the params's description comment
    + `desc` must begin with one the following annotations: {`"[in]"`, `"[out]"`, `"[in,out]"`, `"[nocheck]"`} 
      - `in` is used for params that are read-only; if the param is a pointer, then the memory being pointed to is also read-only
      - `out` is used for params that are write-only; if the param is a pointer, then the memory being pointed to is also write-only
      - `in,out` is used for params that are both read and write; typically this is used for pointers to other data structures that contain both read and write params
      - `nocheck` is used to specify that no additional validation checks will be generated.
    + `desc` may include one the following annotations: {`"[optional]"`, `"[range(start,end)]"`, `"[retain]"`, `"[release]"`, `"[typename(typeVarName)]"`, `"[bounds(offset,size)]"`}
      - `optional` is used for params that are handles or pointers where it is legal for the value to be `nullptr`
      - `range` is used for params that are array pointers to specify the valid range that the is valid to read
        + `start` and `end` must be an ISO-C standard identifier or literal
        + `start` is inclusive and `end` is exclusive
      - `retain` is used for params that are handles or pointers to handles where the function will increment the reference counter associated with the handle(s).
      - `release` is used for params that are handles or pointers to handles where the function will decrement the handle's reference count, potentially leaving it in an invalid state if the reference count reaches zero.
      - `typename` is used to denote the type enum for params that are opaque pointers to values of tagged data types.
      - `bounds` is used for params that are memory objects or USM allocations. It specifies the range within the memory allocation represented by the param that will be accessed by the operation.
        + `offset` and `size` must be an ISO-C standard identifier or literal
        + The sum of `offset` and `size` will be compared against the size of the memory allocation represented by the param.
        + If `offset` and `size` are not both integers they must be of the types `$x_rect_offset` and `$x_rect_region` respectively.
        + If `bounds` is used the operation must also take a parameter of type `$x_queue_handle_t`
    + `type` must be an ISO-C standard identifier
    + `name` must be a unique ISO-C standard identifier
  - A param may take the following optional scalar field: {`init`, `version`}
    + `init` will be used to initialize the C++ function param's value
    + `init` must be an ISO-C standard identifier or literal
    + `version` will be used to define the minimum API version in which the param will appear; `default="1.0"` This will also affect the order in which the param appears within the function.
  - if `class` is specified and the function is not `decl: static`, then the first param **must** be the handle associated with the class
* A function may take the following optional sequence of scalars: {`analogue`}
  - `analogue` will be used as the function's remarks comment
* A function may take the following optional sequence of scalars or scalars to sequences: {`returns`}
  - `returns` will be used as the function's returns comment
  - `returns` must be an etor of `$x_result_t`
  - `returns` defaults are generated by parsing the function's params' description annotations
  - `returns` may contain a sequence of custom validation layer code blocks
* A function may take the following optional field which can be a scalar, a sequence of scalars or scalars to sequences: {`details`}
  - `details` will be used as the function's detailed comment

<table>
<tr><th>YaML</th><th>C</th><th>C++</th><th>Python</th></tr>
<tr><td>

```yml
type: function
desc: "A brief description..."
name: FnName
name: $xClsName
details: 
    - "A more detailed description..."
analogue:
    - another_function
returns:
    - $X_RESULT_ERROR_INVALID_ARGUMENT:
        - "`0 == value`"
    - $X_RESULT_ERROR_OUT_OF_HOST_MEMORY
params:
    - type: $x_cls_handle_t
      name: hClsName
      desc: "[in] handle to class"
    - type: uint32_t
      name: count
      desc: "[in] brief description"
    - type: uint32_t*
      name: values
      desc: "[in][range(0,count)] brief description"
    - type: uint32_t*
      name: result
      desc: "[out] result of function
```
</td>
<td>

```c
/// @brief A brief description...
/// @details
///     - A more detailed description...
/// @remarks
///     _Analogues_
///         - another_function
/// @returns
///     - UR_RESULT_SUCCESS
///     - UR_RESULT_ERROR_INVALID_ARGUMENT
///         + `0 == value`
///     - UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
__ur_api_export ur_result_t __urcall
urClsNameFnName(
    ur_cls_handle_t hClsName,   ///< [in] handle to class
    uint32_t count,             ///< [in] brief description
    uint32_t* values,           ///< [in][range(0,count)] brief description
    uint32_t* result            ///< [out] result of function
    );
```
</td>
<td>

```cpp
namespace ur {
  class ClsName
  {
    /// @brief A brief description...
    /// @details
    ///     - A more detailed description...
    /// @remarks
    ///     _Analogues_
    ///         - another_function
    /// @returns
    ///     - uint32_t: result of function
    /// @throws result_t
    uint32_t __urcall FnName(
        uint32_t count,     ///< [in] brief description
        uint32_t* values    ///< [in][range(0,count)] brief description
        );
```
</td>
<td>

```python
## @brief A brief description...
_urClsNameFnName_t = CFUNCTYPE( ur_result_t, ur_cls_handle_t, c_ulong, POINTER(c_ulong), POINTER(c_ulong) )
```
</td></tr>
</table>

#### type: class
* A class initiates the creation of a C++ class declaration in the specification
* A class requires the following scalar fields: {`desc`, `name`}
  - `desc` will be used as the class's description comment
  - `name` must be a unique ISO-C standard identifier, start with `$` tag, and be PascalCase
* A class may take the following optional scalar fields: {`attribute`, `base`, `owner`, `condition`, `ordinal`, `version`}
  - `attribute` will be used to specify whether the class is a special-type: {`singleton`}
  - `base` will be used to specify the base type of the class
  - `owner` will be used to specify which other class creates this class
  - `condition` will be used as a C/C++ preprocessor `#if` conditional expression
  - `ordinal` will be used to override the default order (in which they appear) the class appears within its section; `default="1000"`
  - `version` will be used to define the minimum API version in which the class will appear; `default="1.0"` This will also affect the order in which the class appears within its section.
* A class requires the following sequence of mappings: {`members`}
  - A member requires the following scalar fields: {`desc`, `type`, `name`}
    + `desc` will be used as the members's description comment
    + `type` must be an ISO-C standard identifier
    + `name` must be a unique ISO-C standard identifier
  - A member may take the following optional scalar field: {`version`}
    + `version` will be used to define the minimum API version in which the member will appear; `default="1.0"` This will also affect the order in which the member appears within the class.
  - The first member must be the handle associated with the class; the `name` must be `"handle"`
  - If `owner` is specified, then the second member must be the pointer to the owner 
  - The next member may be the `_desc_t` used to create the object; the `name` must be `"desc"`
* A class may take the following optional field which can be a scalar, a sequence of scalars or scalars to sequences: {`details`}
  - `details` will be used as the class's detailed comment

<table>
<tr><th>YaML</th><th>C++</th></tr>
<tr><td>

```yml
type: class
desc: "A brief description..."
name: $xClsName
members:
    - type: $x_cls_handle_t
      name: handle
      desc: "brief description"
```
</td>
<td>

```cpp
namespace ur {
    class ClsName
    {
    protected:
        ur_cls_handle_t m_handle;

    public:
        ClsName() = delete;
        ClsName( ur_cls_handle_t handle );

        ~ClsName( void ) = default;

        ClsName( ClsName const& other ) = delete;
        void operator=( ClsName const& other ) = delete;

        ClsName( ClsName&& other ) = delete;
        void operator=( ClsName&& other ) = delete;

        auto getHandle( void ) const { return m_handle; }
    };
```
</td></tr>
<tr><td>

```yml
type: class
desc: "A brief description..."
name: $xClsName2
base: $xClsName
```
</td>
<td>

```cpp
namespace ur {
    class ClsName2 : public ClsName
    {
    public:
        using ClsName::ClsName;

        ~ClsName2( void ) = default;

        ClsName2( ClsName2 const& other ) = delete;
        void operator=( ClsName2 const& other ) = delete;

        ClsName2( ClsName2&& other ) = delete;
        void operator=( ClsName2&& other ) = delete;
    };
```
</td></tr>
</table>

## Extensions
* Each extensions must be defined in a unique `.yml` file
* The extension file must be added to the section being extended; i.e. extensions to core APIs must be added to the `core` folder, etc.
* The extension file must contain a macro that defines the extension name
* The extension file must contain a enum that defines the extension version(s)
* The extension must following the naming convention described in the programming guide
* The extension can add any document type allowed (macro, enum, structure, function, class, etc.)

