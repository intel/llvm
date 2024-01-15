import itertools
import sys
import os
from collections import defaultdict


class TemplatedType:
    """
    Common super class representing a templated type.
    valid_types - A list of the valid types for the instantiation of the template.
    valid_sizes - A list of the valid sizes for the type instantiating the
      template. This is used by vec and marray to restrict their sizes and may be
      None to signify no restrictions.
    """

    def __init__(self, valid_types, valid_sizes):
        self.valid_types = valid_types
        self.valid_sizes = valid_sizes


class Vec(TemplatedType):
    """Class representing a sycl::vec type in SYCL builtins."""

    def __init__(self, valid_types, valid_sizes={1, 2, 3, 4, 8, 16}):
        super().__init__(valid_types, valid_sizes)

    def get_requirements(self, type_name):
        """
        Gets the constexpr requirements of a vector type argument:
        1. The argument must be either a vector or a swizzle.
        2. The element type of the argument must be in valid_types
        3. The number of elements must be in valid_sizes.
        """
        valid_type_str = ", ".join(self.valid_types)
        valid_sizes_str = ", ".join(map(str, self.valid_sizes))
        return [
            f"detail::is_vec_or_swizzle_v<{type_name}>",
            f"detail::is_valid_elem_type_v<{type_name}, {valid_type_str}>",
            f"detail::is_valid_size_v<{type_name}, {valid_sizes_str}>",
        ]

    def __hash__(self):
        return hash(("vec", frozenset(self.valid_types), frozenset(self.valid_sizes)))


class Marray(TemplatedType):
    """Class representing a sycl::marray type in SYCL builtins."""

    def __init__(self, valid_types, valid_sizes=None):
        super().__init__(valid_types, valid_sizes)

    def get_requirements(self, type_name):
        """
        Gets the constexpr requirements of an marray type argument:
        1. The argument must be an marray.
        2. The element type of the argument must be in valid_types
        3. The number of elements must be in valid_sizes. If valid_sizes is None,
           this requirement is excluded.
        """
        valid_type_str = ", ".join(self.valid_types)
        result = [
            f"detail::is_marray_v<{type_name}>",
            f"detail::is_valid_elem_type_v<{type_name}, {valid_type_str}>",
        ]
        if self.valid_sizes:
            valid_sizes_str = ", ".join(map(str, self.valid_sizes))
            result.append(f"detail::is_valid_size_v<{type_name}, {valid_sizes_str}>")
        return result

    def __hash__(self):
        valid_sizes_set = frozenset(self.valid_sizes) if self.valid_sizes else None
        return hash(("marray", frozenset(self.valid_types), valid_sizes_set))


class TemplatedScalar(TemplatedType):
    """
    Represents a templated scalar type, i.e. an alias template that can have any
    of the specified scalar types.
    """

    def __init__(self, valid_types):
        super().__init__(valid_types, None)

    def get_requirements(self, type_name):
        """
        Gets the constexpr requirements of the scalar template type, which is only
        a requirement that the type is one of the types in valid_types.
        """
        valid_type_str = ", ".join(self.valid_types)
        return [f"detail::check_type_in_v<{type_name}, {valid_type_str}>"]

    def __hash__(self):
        return hash(("scalar", frozenset(self.valid_types)))


class MultiPtr(TemplatedType):
    """
    A class representing a templated sycl::multi_ptr type.
    element_type - The type pointed to by the multi_ptr. If both this and
      parent_idx is set, this will instead refer to the element type of the type
      pointed to by the multi_ptr.
    parent_idx - The index of the parent argument.
    """

    def __init__(self, element_type=None, parent_idx=None):
        self.element_type = element_type
        self.parent_idx = parent_idx

    def get_pointed_to_type(self):
        if self.parent_idx is not None and self.element_type is not None:
            return f"detail::change_elements_t<{self.element_type}, T{self.parent_idx}>"
        parent_type = (
            f"T{self.parent_idx}" if self.parent_idx is not None else self.element_type
        )
        return f"detail::simplify_if_swizzle_t<{parent_type}>"

    def get_requirements(self, type_name):
        """
        Gets the constexpr requirements of an multi_ptr type argument:
        1. The argument must be a multi_ptr.
        2. The address space is "writeable".
        3. The type pointed to by the multi_ptr must be the same as the element
             type.
        """
        result = [
            f"detail::is_multi_ptr_v<{type_name}>",
            f"detail::has_writeable_addr_space_v<{type_name}>",
            f"detail::is_valid_elem_type_v<{type_name}, {self.get_pointed_to_type()}>",
        ]
        return result

    def __hash__(self):
        return hash(("multi_ptr", self.element_type))


class RawPtr:
    """
    A class representing a raw pointer type.
    element_type - The type pointed to by the raw pointer.
    TODO: Raw pointer variants kept for compatibility. They are deprecated and
          should be removed.
    """

    def __init__(self, element_type):
        self.element_type = element_type
        self.deprecation_message = "SYCL builtin functions with raw pointer arguments have been deprecated. Please use multi_ptr."

    def __str__(self):
        return f"{self.element_type}*"


class ElementType:
    """
    A class representing the special builtins type for using the element type of
    another argument.
    parent_idx - The index of the parent argument in the corresponding builtin.
    """

    def __init__(self, parent_idx):
        self.parent_idx = parent_idx


class ConversionTraitType:
    """
    A class representing the special cases where a utility trait is applied to
    another argument type to get the corresponding type.
    trait - The name of the trait to use for converting the type.
    parent_idx - The index of the parent argument in the corresponding builtin.
    NOTE: Traits are primarily defined in sycl/include/sycl/builtins_utils.hpp.
    """

    def __init__(self, trait, parent_idx):
        self.trait = trait
        self.parent_idx = parent_idx


class InstantiatedTemplatedArg:
    """
    A class representing an instantiated template argument, i.e. a template
    argument with a given type and an assigned template name.
    template_name - The name of the argument's type in the template function.
    template_type - The associated type picked from the valid types in the builtin
      definition.
    """

    def __init__(self, template_name, templated_type):
        self.template_name = template_name
        self.templated_type = templated_type

    def __str__(self):
        return self.template_name


class InstantiatedTemplatedReturnType:
    """
    A class representing an instantiated template return type, i.e. an association
    between the return type of a builtin and a related template argument.
    related_arg_type - The related template argument type.
    template_type - The associated type picked from the valid types in the builtin
      definition.
    """

    def __init__(self, related_arg_type, templated_type):
        self.related_arg_type = related_arg_type
        self.templated_type = templated_type

    def __str__(self):
        if isinstance(self.templated_type, Vec):
            # Vectors need conversion as their related type may be a swizzle.
            return f"detail::simplify_if_swizzle_t<{self.related_arg_type}>"
        return str(self.related_arg_type)


class InstantiatedElementType:
    """
    A class representing an instantiated version of ElementType containing the
    referenced type.
    referenced_type - The referenced type.
    parent_idx - The index of the parent argument in the corresponding builtin.
    """

    def __init__(self, referenced_type, parent_idx):
        self.referenced_type = referenced_type
        self.parent_idx = parent_idx

    def __str__(self):
        if isinstance(self.referenced_type, InstantiatedTemplatedArg) or isinstance(
            self.referenced_type, InstantiatedTemplatedReturnType
        ):
            return f"detail::get_elem_type_t<{self.referenced_type}>"
        return self.referenced_type


class InstantiatedConversionTraitType:
    """
    A class representing an instantiated version of ConversionTraitType containing
    the parent type.
    parent_type - The parent type.
    trait - The name of the trait to use for converting the type.
    parent_idx - The index of the parent argument in the corresponding builtin.
    """

    def __init__(self, parent_type, trait, parent_idx):
        self.parent_type = parent_type
        self.trait = trait
        self.parent_idx = parent_idx

    def __str__(self):
        return f"detail::{self.trait}<{self.parent_type}>"


### GENTYPE DEFINITIONS

vfloatn = [Vec(["float"])]
vfloat3or4 = [Vec(["float"], {3, 4})]
mfloatn = [Marray(["float"])]
mfloat3or4 = [Marray(["float"], {3, 4})]
genfloatf = ["float", Vec(["float"]), Marray(["float"])]

vdoublen = [Vec(["double"])]
vdouble3or4 = [Vec(["double"], {3, 4})]
mdoublen = [Marray(["double"])]
mdouble3or4 = [Marray(["double"], {3, 4})]

vhalfn = [Vec(["half"])]
vhalf3or4 = [Vec(["half"], {3, 4})]
mhalfn = [Marray(["half"])]
mhalf3or4 = [Marray(["half"], {3, 4})]

genfloat = [
    "float",
    "double",
    "half",
    Vec(["float", "double", "half"]),
    Marray(["float", "double", "half"]),
]
vgenfloat = [Vec(["float", "double", "half"])]
sgenfloat = ["float", "double", "half"]
mgenfloat = [Marray(["float", "double", "half"])]

vgeofloat = [Vec(["float"], {1, 2, 3, 4})]
vgeodouble = [Vec(["double"], {1, 2, 3, 4})]
vgeohalf = [Vec(["half"], {1, 2, 3, 4})]
mgeofloat = [Marray(["float"], {1, 2, 3, 4})]
mgeodouble = [Marray(["double"], {1, 2, 3, 4})]
mgeohalf = [Marray(["half"], {1, 2, 3, 4})]
gengeofloat = ["float", Vec(["float"], {1, 2, 3, 4}), Marray(["float"], {1, 2, 3, 4})]
gengeodouble = [
    "double",
    Vec(["double"], {1, 2, 3, 4}),
    Marray(["double"], {1, 2, 3, 4}),
]
gengeohalf = ["half", Vec(["half"], {1, 2, 3, 4}), Marray(["half"], {1, 2, 3, 4})]

vint8n = [
    Vec(["int8_t", "char", "signed char"])
]  # Fundamental integer types non-standard. Deprecated.
vint16n = [
    Vec(["int16_t", "short"])
]  # Fundamental integer types non-standard. Deprecated.
vint32n = [
    Vec(["int32_t", "int"])
]  # Fundamental integer types non-standard. Deprecated.
vint64n = [
    Vec(["int64_t", "long long"])
]  # Fundamental integer types non-standard. Deprecated.
vuint8n = [
    Vec(["uint8_t", "unsigned char"])
]  # Fundamental integer types non-standard. Deprecated.
vuint16n = [
    Vec(["uint16_t", "unsigned short"])
]  # Fundamental integer types non-standard. Deprecated.
vuint32n = [
    Vec(["uint32_t", "unsigned int"])
]  # Fundamental integer types non-standard. Deprecated.
vuint64n = [
    Vec(["uint64_t", "unsigned long long"])
]  # Fundamental integer types non-standard. Deprecated.

# vuint<N>n name is taken, use "_fixed" suffix.
vuint16n_fixed = [Vec(["uint16_t"])]
vuint32n_fixed = [Vec(["uint32_t"])]
vuint64n_fixed = [Vec(["uint64_t"])]

mint8n = [Marray(["int8_t"])]
mint16n = [Marray(["int16_t"])]
mint32n = [Marray(["int32_t"])]
muint8n = [Marray(["uint8_t"])]
muint16n = [Marray(["uint16_t"])]
muint32n = [Marray(["uint32_t"])]
muint64n = [Marray(["uint64_t"])]
mintn = [Marray(["int"])]
mushortn = [Marray(["unsigned short"])]
muintn = [Marray(["unsigned int"])]
mulongn = [Marray(["unsigned long", "unsigned long long"])]
mbooln = [Marray(["bool"])]

geninteger = [
    "char",
    "signed char",
    "short",
    "int",
    "long",
    "long long",
    "unsigned char",
    "unsigned short",
    "unsigned int",
    "unsigned long",
    "unsigned long long",
    Vec(
        [
            "int8_t",
            "int16_t",
            "int32_t",
            "int64_t",
            "uint8_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
            "char",
            "signed char",
            "short",
            "int",
            "long",
            "long long",
            "unsigned char",
            "unsigned short",
            "unsigned int",
            "unsigned long",
            "unsigned long long",
        ]
    ),  # Fundamental integer types non-standard. Deprecated.
    Marray(
        [
            "char",
            "signed char",
            "short",
            "int",
            "long",
            "long long",
            "unsigned char",
            "unsigned short",
            "unsigned int",
            "unsigned long",
            "unsigned long long",
        ]
    ),
]
sigeninteger = ["char", "signed char", "short", "int", "long", "long long"]
vigeninteger = [
    Vec(
        [
            "int8_t",
            "int16_t",
            "int32_t",
            "int64_t",
            "char",
            "signed char",
            "short",
            "int",
            "long",
            "long long",
        ]
    )
]  # Fundamental integer types non-standard. Deprecated.
migeninteger = [Marray(["char", "signed char", "short", "int", "long", "long long"])]
igeninteger = [
    "char",
    "signed char",
    "short",
    "int",
    "long",
    "long long",
    Vec(
        [
            "int8_t",
            "int16_t",
            "int32_t",
            "int64_t",
            "char",
            "signed char",
            "short",
            "int",
            "long",
            "long long",
        ]
    ),  # Fundamental integer types non-standard. Deprecated.
    Marray(["char", "signed char", "short", "int", "long", "long long"]),
]
vugeninteger = [
    Vec(
        [
            "uint8_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
            "unsigned char",
            "unsigned short",
            "unsigned int",
            "unsigned long",
            "unsigned long long",
        ]
    )
]  # Non-standard. Deprecated.
ugeninteger = [
    Vec(
        [
            "uint8_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
            "unsigned char",
            "unsigned short",
            "unsigned int",
            "unsigned long",
            "unsigned long long",
        ]
    ),  # Non-standard. Deprecated.
    Marray(
        [
            "unsigned char",
            "unsigned short",
            "unsigned int",
            "unsigned long",
            "unsigned long long",
        ]
    ),
    "unsigned char",
    "unsigned short",
    "unsigned int",
    "unsigned long",
    "unsigned long long",
]
igenint32 = ["int32_t", Vec(["int32_t"]), Marray(["int32_t"])]
ugenint32 = ["uint32_t", Vec(["uint32_t"]), Marray(["uint32_t"])]
genint32 = [
    "int32_t",
    "uint32_t",
    Vec(["int32_t", "uint32_t"]),
    Marray(["int32_t", "uint32_t"]),
]

sgentype = [
    "char",
    "signed char",
    "short",
    "int",
    "long",
    "long long",
    "unsigned char",
    "unsigned short",
    "unsigned int",
    "unsigned long",
    "unsigned long long",
    "float",
    "double",
    "half",
]
vgentype = [
    Vec(
        [
            "int8_t",
            "int16_t",
            "int32_t",
            "int64_t",
            "uint8_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
            "float",
            "double",
            "half",
            "char",
            "signed char",
            "short",
            "int",
            "long",
            "long long",
            "unsigned char",
            "unsigned short",
            "unsigned int",
            "unsigned long",
            "unsigned long long",
        ]
    )
]  # Fundamental integer types non-standard. Deprecated.
mgentype = [
    Marray(
        [
            "char",
            "signed char",
            "short",
            "int",
            "long",
            "long long",
            "unsigned char",
            "unsigned short",
            "unsigned int",
            "unsigned long",
            "unsigned long long",
            "float",
            "double",
            "half",
        ]
    )
]

intptr = [MultiPtr("int"), RawPtr("int")]
floatptr = [MultiPtr("float"), RawPtr("float")]
doubleptr = [MultiPtr("double"), RawPtr("double")]
halfptr = [MultiPtr("half"), RawPtr("half")]
rawvgenfloatptr0 = [RawPtr(Vec(["float", "double", "half"]))]
ptr0 = [MultiPtr(parent_idx=0)]
intnptr0 = [MultiPtr("int", parent_idx=0)]
vint32ptr0 = [MultiPtr("int32_t", parent_idx=0), RawPtr(Vec(["int32_t"]))]

# To help resolve template arguments, these are given the index of their parent
# argument.
elementtype0 = [ElementType(0)]
samesizesignedint0 = [ConversionTraitType("same_size_signed_int_t", 0)]
samesizeunsignedint0 = [ConversionTraitType("same_size_unsigned_int_t", 0)]
intelements0 = [ConversionTraitType("int_elements_t", 0)]
boolelements0 = [ConversionTraitType("bool_elements_t", 0)]
upsampledint0 = [ConversionTraitType("upsampled_int_t", 0)]
nanreturn0 = [ConversionTraitType("nan_return_unswizzled_t", 0)]

# Map of builtin type group names and the associated types.
builtin_types = {
    "vfloatn": vfloatn,
    "vfloat3or4": vfloat3or4,
    "mfloatn": mfloatn,
    "mfloat3or4": mfloat3or4,
    "genfloatf": genfloatf,
    "vdoublen": vdoublen,
    "vdouble3or4": vdouble3or4,
    "mdoublen": mdoublen,
    "mdouble3or4": mdouble3or4,
    "vhalfn": vhalfn,
    "vhalf3or4": vhalf3or4,
    "mhalfn": mhalfn,
    "mhalf3or4": mhalf3or4,
    "genfloat": genfloat,
    "vgenfloat": vgenfloat,
    "sgenfloat": sgenfloat,
    "mgenfloat": mgenfloat,
    "vgeofloat": vgeofloat,
    "vgeodouble": vgeodouble,
    "vgeohalf": vgeohalf,
    "mgeofloat": mgeofloat,
    "mgeodouble": mgeodouble,
    "mgeohalf": mgeohalf,
    "gengeofloat": gengeofloat,
    "gengeodouble": gengeodouble,
    "gengeohalf": gengeohalf,
    "vint8n": vint8n,
    "vint16n": vint16n,
    "vint32n": vint32n,
    "vint64n": vint64n,
    "vuint8n": vuint8n,
    "vuint16n": vuint16n,
    "vuint32n": vuint32n,
    "vuint64n": vuint64n,
    "vuint16n_fixed": vuint16n_fixed,
    "vuint32n_fixed": vuint32n_fixed,
    "vuint64n_fixed": vuint64n_fixed,
    "mint8n": mint8n,
    "mint16n": mint16n,
    "mint32n": mint32n,
    "muint8n": muint8n,
    "muint16n": muint16n,
    "muint32n": muint32n,
    "muint64n": muint64n,
    "mintn": mintn,
    "mushortn": mushortn,
    "muintn": muintn,
    "mulongn": mulongn,
    "mbooln": mbooln,
    "geninteger": geninteger,
    "sigeninteger": sigeninteger,
    "vigeninteger": vigeninteger,
    "migeninteger": migeninteger,
    "igeninteger": igeninteger,
    "vugeninteger": vugeninteger,
    "ugeninteger": ugeninteger,
    "igenint32": igenint32,
    "ugenint32": ugenint32,
    "sgentype": sgentype,
    "vgentype": vgentype,
    "mgentype": mgentype,
    "intptr": intptr,
    "floatptr": floatptr,
    "doubleptr": doubleptr,
    "halfptr": halfptr,
    "rawvgenfloatptr0": rawvgenfloatptr0,
    "ptr0": ptr0,
    "intnptr0": intnptr0,
    "vint32nptr0": vint32ptr0,
    "elementtype0": elementtype0,
    "samesizesignedint0": samesizesignedint0,
    "samesizeunsignedint0": samesizeunsignedint0,
    "intelements0": intelements0,
    "upsampledint0": upsampledint0,
    "boolelements0": boolelements0,
    "nanreturn0": nanreturn0,
    "char": ["char"],
    "signed char": ["signed char"],
    "short": ["short"],
    "int": ["int"],
    "long": ["long"],
    "long long": ["long long"],
    "unsigned char": ["unsigned char"],
    "unsigned short": ["unsigned short"],
    "unsigned int": ["unsigned int"],
    "unsigned long": ["unsigned long"],
    "unsigned long long": ["unsigned long long"],
    "float": ["float"],
    "double": ["double"],
    "half": ["half"],
    "int8_t": ["int8_t"],
    "int16_t": ["int16_t"],
    "int32_t": ["int32_t"],
    "int64_t": ["int64_t"],
    "uint8_t": ["uint8_t"],
    "uint16_t": ["uint16_t"],
    "uint32_t": ["uint32_t"],
    "uint64_t": ["uint64_t"],
    "bool": ["bool"],
}

### BUILTINS


def find_first_template_arg(arg_types):
    """Finds the first templated argument in a type list."""
    for arg_type in arg_types:
        if isinstance(arg_type, InstantiatedTemplatedArg):
            return arg_type
    return None


def is_marray_arg(arg_type):
    """Returns true if the argument type is a templated marray."""
    return isinstance(arg_type, InstantiatedTemplatedArg) and isinstance(
        arg_type.templated_type, Marray
    )


def is_vec_arg(arg_type):
    """Returns true if the argument type is a templated vector."""
    return isinstance(arg_type, InstantiatedTemplatedArg) and isinstance(
        arg_type.templated_type, Vec
    )


def is_multi_ptr_arg(arg_type):
    """Returns true if the argument type is a templated multi_ptr."""
    return isinstance(arg_type, InstantiatedTemplatedArg) and isinstance(
        arg_type.templated_type, MultiPtr
    )


def convert_arg_name(arg_type, arg_name):
    """
    Converts an argument name to a valid argument to be passed to another
    function. For vector arguments this is done by converting the argument into
    the associated vector type, needed for swizzles.
    """
    if is_vec_arg(arg_type):
        return f"detail::simplify_if_swizzle_t<{arg_type}>({arg_name})"
    return arg_name


def get_invoke_args(arg_types, arg_names, convert_args=[]):
    """
    Gets the arguments to be used inside builtins when calling the implementation
    invoke functions.
    """
    result = list(map(convert_arg_name, arg_types, arg_names))
    for arg_idx, type_conv in convert_args:
        # type_conv is either an index or a conversion function/type.
        conv = type_conv if isinstance(type_conv, str) else arg_types[type_conv]
        if is_vec_arg(conv):
            # If the conversion is to a vector template argument, it could also be
            # a swizzle. Since we cannot convert most types to swizzles, we make
            # sure to make the conversion to the corresponding vector type instead.
            conv = f"detail::simplify_if_swizzle_t<{conv}>"
        result[arg_idx] = f"{conv}({result[arg_idx]})"
    return result


class DefCommon:
    """
    Common super class representing a builtin definition.
    return_type - The name of the type group of the return type. Must match a key
      in builtin_types.
    arg_types - The names of the type group of each argument in the builtin.
    invoke_name - The implementation invoke function name associated with the
      builtin. For scalar and vector builtins this is prefixed by
      __sycl_std::__invoke_.
    custom_invoke - A function object to be used for generating a custom body for
      the builtin.
    size_alias - A name to use as an alias for the size associated with vector and
      marray arguments.
    marray_use_loop - A bool specifying if marray builtins from this definition
      should use a loop-based implementation instead of the default vectorized
      implementation used when possible.
    template_scalar_args - A bool specifying if the builtin should combine the
      scalar arguments into common template types.
    deprecation_message - A message that will appear in a declaration warning.
    """

    def __init__(
        self,
        return_type,
        arg_types,
        invoke_name,
        invoke_prefix,
        custom_invoke,
        size_alias,
        marray_use_loop,
        template_scalar_args,
        deprecation_message="",
    ):
        self.return_type = return_type
        self.arg_types = arg_types
        self.invoke_name = invoke_name
        self.invoke_prefix = invoke_prefix
        self.custom_invoke = custom_invoke
        self.size_alias = size_alias
        self.marray_use_loop = marray_use_loop
        self.template_scalar_args = template_scalar_args
        self.deprecation_message = deprecation_message

    def require_size_alias(self, alternative_name, marray_type):
        """
        Requires that a size alias is specified in the function body. This returns
        the size alias name and any needed initialization of it. If the size alias
        hasn't been explicitly requested in the definition, one will be generated
        using the alternative_name.
        """
        if not self.size_alias:
            # If there isn't a size alias defined, we add one.
            return (
                alternative_name,
                f"  constexpr size_t {alternative_name} = detail::num_elements<{marray_type.template_name}>::value;\n",
            )
        return (self.size_alias, "")

    def convert_loop_arg(self, arg_type, arg_name):
        """
        Converts arguments in an marray loop builtin implementation to scalars.
        The variable I must be defined in the scope where the generated arguments
        are used and indicate the index of the used element.
        """
        if is_multi_ptr_arg(arg_type):
            pointed_to_type = arg_type.templated_type.get_pointed_to_type()
            return f"address_space_cast<{arg_type}::address_space, detail::get_multi_ptr_decoration_v<{arg_type}>, detail::get_elem_type_t<{pointed_to_type}>>(&(*{arg_name})[I])"
        if is_marray_arg(arg_type):
            return str(arg_name) + "[I]"
        return str(arg_name)

    def get_marray_loop_invoke_body(
        self,
        namespaced_builtin_name,
        return_type,
        arg_types,
        arg_names,
        first_marray_type,
    ):
        """Generated the body of a loop-based implementation of marray builtins."""
        result = ""
        args = [
            self.convert_loop_arg(arg_type, arg_name)
            for arg_type, arg_name in zip(arg_types, arg_names)
        ]
        joined_args = ", ".join(args)
        (size_alias, size_alias_init) = self.require_size_alias("N", first_marray_type)
        result += size_alias_init
        return (
            result
            + f"""  {return_type} Res;
  for (int I = 0; I < {size_alias}; ++I)
    Res[I] = {namespaced_builtin_name}({joined_args});
  return Res;"""
        )

    def get_marray_vec_cast_invoke_body(
        self,
        namespaced_builtin_name,
        return_type,
        arg_types,
        arg_names,
        first_marray_type,
    ):
        """
        Generates the body of a direct vector cast implementation for marray
        builtins.
        """
        result = ""
        vec_cast_args = [
            f"detail::to_vec({arg_name})" if is_marray_arg(arg_type) else str(arg_name)
            for arg_type, arg_name in zip(arg_types, arg_names)
        ]
        joined_vec_cast_args = ", ".join(vec_cast_args)
        vec_call = f"{namespaced_builtin_name}({joined_vec_cast_args})"
        if isinstance(return_type, InstantiatedTemplatedReturnType):
            # Convert the vec call result to marray.
            vec_call = f"detail::to_marray({vec_call})"
        return result + f"  return {vec_call};"

    def get_marray_vectorized_invoke_body(
        self,
        namespaced_builtin_name,
        return_type,
        arg_types,
        arg_names,
        first_marray_type,
    ):
        """Generates the body of a vectorized implementation for marray builtins."""
        result = ""
        (size_alias, size_alias_init) = self.require_size_alias("N", first_marray_type)
        result += size_alias_init
        # Adjust arguments for partial results and the remaining work at the end.
        imm_args = []
        rem_args = []
        for arg_type, arg_name in zip(arg_types, arg_names):
            is_marray = is_marray_arg(arg_type)
            imm_args.append(
                f"detail::to_vec2({arg_name}, I * 2)" if is_marray else arg_name
            )
            rem_args.append(f"{arg_name}[{size_alias} - 1]" if is_marray else arg_name)
        joined_imm_args = ", ".join(imm_args)
        joined_rem_args = ", ".join(rem_args)
        return (
            result
            + f"""  {return_type} Res;
  for (size_t I = 0; I < {size_alias} / 2; ++I) {{
    auto PartialRes = {namespaced_builtin_name}({joined_imm_args});
    std::memcpy(&Res[I * 2], &PartialRes, sizeof(decltype(PartialRes)));
  }}
  if ({size_alias} % 2)
    Res[{size_alias} - 1] = {namespaced_builtin_name}({joined_rem_args});
  return Res;"""
        )

    def get_marray_invoke_body(
        self,
        namespaced_builtin_name,
        return_type,
        arg_types,
        arg_names,
        first_marray_type,
    ):
        """Generates the body of an marray builtin."""
        # If the associated marray types have restriction on their sizes, we assume
        # they can be converted directly to vector.
        if first_marray_type.templated_type.valid_sizes:
            return self.get_marray_vec_cast_invoke_body(
                namespaced_builtin_name,
                return_type,
                arg_types,
                arg_names,
                first_marray_type,
            )
        # If there is a pointer argument, we need to use the simple loop solution.
        if self.marray_use_loop or any(
            [is_multi_ptr_arg(arg_type) for arg_type in arg_types]
        ):
            return self.get_marray_loop_invoke_body(
                namespaced_builtin_name,
                return_type,
                arg_types,
                arg_names,
                first_marray_type,
            )
        # Otherwise, we vectorize the body.
        return self.get_marray_vectorized_invoke_body(
            namespaced_builtin_name,
            return_type,
            arg_types,
            arg_names,
            first_marray_type,
        )

    def get_invoke_body(
        self, builtin_name, namespace, invoke_name, return_type, arg_types, arg_names
    ):
        """Generates the body of a builtin."""
        for arg_type in arg_types:
            if is_marray_arg(arg_type):
                namespaced_builtin_name = (
                    f"{namespace}::{builtin_name}" if namespace else builtin_name
                )
                return self.get_marray_invoke_body(
                    namespaced_builtin_name, return_type, arg_types, arg_names, arg_type
                )
        return self.get_scalar_vec_invoke_body(
            invoke_name, return_type, arg_types, arg_names
        )

    def get_invoke(self, builtin_name, namespace, return_type, arg_types, arg_names):
        """
        Generates the builtin defined by this instance, using the specific types and
        names.
        """
        if self.custom_invoke:
            return self.custom_invoke(return_type, arg_types, arg_names)
        invoke_name = self.invoke_name if self.invoke_name else builtin_name
        result = ""
        if self.size_alias:
            template_arg = find_first_template_arg(arg_types)
            if template_arg:
                result += f"  constexpr size_t {self.size_alias} = detail::num_elements<{template_arg.template_name}>::value;"
        return result + self.get_invoke_body(
            builtin_name, namespace, invoke_name, return_type, arg_types, arg_names
        )


class Def(DefCommon):
    """
    A class representing a builtin definition.
    fast_math_invoke_name - Similar to invoke_name, but will only be used for
      valid types when fast math is enabled.
    fast_math_custom_invoke - Similar to custom_invoke,  but will only be used for
      valid types when fast math is enabled.
    convert_args - A list of either strings or tuples specifying how to convert
      arguments when calling the implementation invocation function.
    NOTE: For additional members, see DefCommon.
    """

    def __init__(
        self,
        return_type,
        arg_types,
        invoke_name=None,
        invoke_prefix="",
        custom_invoke=None,
        fast_math_invoke_name=None,
        fast_math_custom_invoke=None,
        convert_args=[],
        size_alias=None,
        marray_use_loop=False,
        template_scalar_args=False,
        deprecation_message=None,
    ):
        super().__init__(
            return_type,
            arg_types,
            invoke_name,
            invoke_prefix,
            custom_invoke,
            size_alias,
            marray_use_loop,
            template_scalar_args,
            deprecation_message,
        )
        self.fast_math_invoke_name = fast_math_invoke_name
        self.fast_math_custom_invoke = fast_math_custom_invoke
        # List of tuples with mappings for arguments to cast to argument types.
        # First element in a tuple is the index of the argument to cast and the
        # second element is the index of the argument type to convert to.
        # Alternatively, the second element can be a string representation of the
        # conversion function or type.
        self.convert_args = convert_args

    def get_scalar_vec_invoke_body(
        self, invoke_name, return_type, arg_types, arg_names
    ):
        """Generates the body of scalar and vector builtins."""
        invoke_args = get_invoke_args(arg_types, arg_names, self.convert_args)
        result = ""
        if self.fast_math_invoke_name or self.fast_math_custom_invoke:
            result += f"  if constexpr (detail::use_fast_math_v<{arg_types[0]}>) {{"
            if self.fast_math_custom_invoke:
                result += self.fast_math_custom_invoke(
                    return_type, arg_types, arg_names
                )
            else:
                result += f'    return __sycl_std::__invoke_{self.fast_math_invoke_name}<{return_type}>({(", ".join(invoke_args))});'
            result += "}\n"
        return (
            result
            + f'  return __sycl_std::__invoke_{self.invoke_prefix}{invoke_name}<{return_type}>({(", ".join(invoke_args))});'
        )


class RelDef(DefCommon):
    """
    A class representing a relational builtin definition.
    NOTE: For members, see DefCommon.
    """

    def __init__(
        self,
        return_type,
        arg_types,
        invoke_name=None,
        invoke_prefix="",
        custom_invoke=None,
        template_scalar_args=False,
    ):
        # NOTE: Relational builtins never use the vectorized solution as the vectors
        #       are likely to use values larger than bool.
        super().__init__(
            return_type,
            arg_types,
            invoke_name,
            invoke_prefix,
            custom_invoke,
            None,
            True,
            template_scalar_args,
        )

    #
    def get_scalar_vec_invoke_body(
        self, invoke_name, return_type, arg_types, arg_names
    ):
        """Generates the body of scalar and vector builtins."""
        if self.custom_invoke:
            return self.custom_invoke(return_type, arg_types, arg_names)
        invoke_args = ", ".join(get_invoke_args(arg_types, arg_names))
        return f"  return detail::RelConverter<{return_type}>::apply(__sycl_std::__invoke_{self.invoke_prefix}{invoke_name}<detail::internal_rel_ret_t<{return_type}>>({invoke_args}));"


def get_custom_unsigned_to_signed_scalar_invoke(invoke_name):
    """
    Creates a function for generating the custom body for invocations returning
    an unsigned scalar value, which will in turn be converted to a signed value.
    """
    return (
        lambda return_type, _, arg_names: f'return static_cast<{return_type}>(__sycl_std::__invoke_{invoke_name}<detail::make_unsigned_t<{return_type}>>({" ,".join(arg_names)}));'
    )


def get_custom_unsigned_to_signed_vec_invoke(invoke_name):
    """
    Creates a function for generating the custom body for invocations returning
    an unsigned scalar value, which will in turn be converted to a signed value.
    """
    return (
        lambda return_type, arg_types, arg_names: f'return __sycl_std::__invoke_{invoke_name}<detail::make_unsigned_t<{return_type}>>({" ,".join(get_invoke_args(arg_types, arg_names))}).template convert<detail::get_elem_type_t<{return_type}>>();'
    )


def get_custom_any_all_vec_invoke(invoke_name):
    """
    Creates a function for generating the custom body for either `any` or `all`
    scalar and vector builtins.
    """
    return (
        lambda _, arg_types, arg_names: f"""  using VecT = detail::simplify_if_swizzle_t<{arg_types[0]}>;
  return detail::rel_sign_bit_test_ret_t<detail::simplify_if_swizzle_t<VecT>>(
      __sycl_std::__invoke_{invoke_name}<detail::rel_sign_bit_test_ret_t<VecT>>(
          detail::rel_sign_bit_test_arg_t<VecT>({get_invoke_args(arg_types, arg_names)[0]})));"""
    )


def custom_bool_select_invoke(return_type, _, arg_names):
    """Generates the custom body for `select` with the last argument being bool."""
    return f"""  return __sycl_std::__invoke_select<{return_type}>(
      {arg_names[0]}, {arg_names[1]}, static_cast<detail::get_select_opencl_builtin_c_arg_type<{return_type}>>({arg_names[2]}));"""


def get_custom_any_all_marray_invoke(builtin):
    """
    Creates a function for generating the custom body for either `any` or `all`
    marray builtins.
    """
    return (
        lambda _, arg_types, arg_names: f"  return std::{builtin}_of({arg_names[0]}.begin(), {arg_names[0]}.end(), [](detail::get_elem_type_t<{arg_types[0]}> X) {{ return {builtin}(X); }});"
    )


def custom_fast_math_sincos_invoke(return_type, _, arg_names):
    """
    Generates the custom body for `sincos` in fast-math mode.
    This is a performance optimization to ensure that sincos isn't slower than a
    pair of sin/cos executed separately. Theoretically, calling non-native sincos
    might be faster than calling native::sin plus native::cos separately and we'd
    need some kind of cost model to make the right decision (and move this
    entirely to the JIT/AOT compilers). However, in practice, this simpler
    solution seems to work just fine and matches how sin/cos above are optimized
    for the fast math path.
    """
    return f"""    *{arg_names[1]} = __sycl_std::__invoke_native_cos<{return_type}>({arg_names[0]});
    return __sycl_std::__invoke_native_sin<{return_type}>({arg_names[0]});"""


def custom_nan_invoke(return_type, arg_types, arg_names):
    """
    Generates the custom body for the `nan` function.
    """
    return f"""  using unswizzled_arg_t = detail::simplify_if_swizzle_t<{arg_types[0]}>;
  return __sycl_std::__invoke_nan<{return_type}>(
    detail::convert_data_type<unswizzled_arg_t, detail::nan_argument_base_t<unswizzled_arg_t>>()({arg_names[0]}));"""


# List of all builtins definitions in the sycl namespace.
sycl_builtins = {  # Math functions
    "acos": [Def("genfloat", ["genfloat"])],
    "acosh": [Def("genfloat", ["genfloat"])],
    "acospi": [Def("genfloat", ["genfloat"])],
    "asin": [Def("genfloat", ["genfloat"])],
    "asinh": [Def("genfloat", ["genfloat"])],
    "asinpi": [Def("genfloat", ["genfloat"])],
    "atan": [Def("genfloat", ["genfloat"])],
    "atan2": [Def("genfloat", ["genfloat", "genfloat"])],
    "atanh": [Def("genfloat", ["genfloat"])],
    "atanpi": [Def("genfloat", ["genfloat"])],
    "atan2pi": [Def("genfloat", ["genfloat", "genfloat"])],
    "cbrt": [Def("genfloat", ["genfloat"])],
    "ceil": [Def("genfloat", ["genfloat"])],
    "copysign": [Def("genfloat", ["genfloat", "genfloat"])],
    "cos": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_cos")],
    "cosh": [Def("genfloat", ["genfloat"])],
    "cospi": [Def("genfloat", ["genfloat"])],
    "erfc": [Def("genfloat", ["genfloat"])],
    "erf": [Def("genfloat", ["genfloat"])],
    "exp": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_exp")],
    "exp2": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_exp2")],
    "exp10": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_exp10")],
    "expm1": [Def("genfloat", ["genfloat"])],
    "fabs": [Def("genfloat", ["genfloat"])],
    "fdim": [Def("genfloat", ["genfloat", "genfloat"])],
    "floor": [Def("genfloat", ["genfloat"])],
    "fma": [Def("genfloat", ["genfloat", "genfloat", "genfloat"])],
    "fmax": [
        Def("genfloat", ["genfloat", "genfloat"]),
        Def("vgenfloat", ["vgenfloat", "elementtype0"], convert_args=[(1, 0)]),
        Def("mgenfloat", ["mgenfloat", "elementtype0"]),
    ],
    "fmin": [
        Def("genfloat", ["genfloat", "genfloat"]),
        Def("vgenfloat", ["vgenfloat", "elementtype0"], convert_args=[(1, 0)]),
        Def("mgenfloat", ["mgenfloat", "elementtype0"]),
    ],
    "fmod": [Def("genfloat", ["genfloat", "genfloat"])],
    "fract": [
        Def("vgenfloat", ["vgenfloat", "rawvgenfloatptr0"]),
        Def("vgenfloat", ["vgenfloat", "ptr0"]),
        Def("mgenfloat", ["mgenfloat", "ptr0"]),
        Def("float", ["float", "floatptr"]),
        Def("double", ["double", "doubleptr"]),
        Def("half", ["half", "halfptr"]),
    ],
    "frexp": [
        Def("vgenfloat", ["vgenfloat", "vint32nptr0"]),
        Def("mgenfloat", ["mgenfloat", "intnptr0"], marray_use_loop=True),
        Def("float", ["float", "intptr"]),
        Def("double", ["double", "intptr"]),
        Def("half", ["half", "intptr"]),
    ],
    "hypot": [Def("genfloat", ["genfloat", "genfloat"])],
    "ilogb": [
        Def("intelements0", ["vgenfloat"]),
        Def("intelements0", ["mgenfloat"]),
        Def("int", ["float"]),
        Def("int", ["double"]),
        Def("int", ["half"]),
    ],
    "ldexp": [
        Def("vgenfloat", ["vgenfloat", "vint32n"]),
        Def(
            "vgenfloat",
            ["vgenfloat", "int"],
            convert_args=[(1, "vec<int, N>")],
            size_alias="N",
        ),
        Def("mgenfloat", ["mgenfloat", "mintn"], marray_use_loop=True),
        Def("mgenfloat", ["mgenfloat", "int"]),
        Def("float", ["float", "int"]),
        Def("double", ["double", "int"]),
        Def("half", ["half", "int"]),
    ],
    "lgamma": [Def("genfloat", ["genfloat"])],
    "lgamma_r": [
        Def("vgenfloat", ["vgenfloat", "vint32nptr0"]),
        Def("mgenfloat", ["mgenfloat", "intnptr0"]),
        Def("float", ["float", "intptr"]),
        Def("double", ["double", "intptr"]),
        Def("half", ["half", "intptr"]),
    ],
    "log": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_log")],
    "log2": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_log2")],
    "log10": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_log10")],
    "log1p": [Def("genfloat", ["genfloat"])],
    "logb": [Def("genfloat", ["genfloat"])],
    "mad": [Def("genfloat", ["genfloat", "genfloat", "genfloat"])],
    "maxmag": [Def("genfloat", ["genfloat", "genfloat"])],
    "minmag": [Def("genfloat", ["genfloat", "genfloat"])],
    "modf": [
        Def("vgenfloat", ["vgenfloat", "rawvgenfloatptr0"]),
        Def("vgenfloat", ["vgenfloat", "ptr0"]),
        Def("mgenfloat", ["mgenfloat", "ptr0"]),
        Def("float", ["float", "floatptr"]),
        Def("double", ["double", "doubleptr"]),
        Def("half", ["half", "halfptr"]),
    ],
    "nan": [
        Def("nanreturn0", ["uint16_t"], custom_invoke=custom_nan_invoke),
        Def("nanreturn0", ["vuint16n_fixed"], custom_invoke=custom_nan_invoke),
        Def("nanreturn0", ["muint16n"], marray_use_loop=True),
        Def("nanreturn0", ["uint32_t"], custom_invoke=custom_nan_invoke),
        Def("nanreturn0", ["vuint32n_fixed"], custom_invoke=custom_nan_invoke),
        Def("nanreturn0", ["muint32n"], marray_use_loop=True),
        Def("nanreturn0", ["uint64_t"], custom_invoke=custom_nan_invoke),
        Def("nanreturn0", ["vuint64n_fixed"], custom_invoke=custom_nan_invoke),
        Def("nanreturn0", ["muint64n"], marray_use_loop=True),
    ],
    "nextafter": [Def("genfloat", ["genfloat", "genfloat"])],
    "pow": [Def("genfloat", ["genfloat", "genfloat"])],
    "pown": [
        Def("vgenfloat", ["vgenfloat", "vint32n"]),
        Def("mgenfloat", ["mgenfloat", "mintn"], marray_use_loop=True),
        Def("float", ["float", "int"]),
        Def("double", ["double", "int"]),
        Def("half", ["half", "int"]),
    ],
    "powr": [
        Def("genfloat", ["genfloat", "genfloat"], fast_math_invoke_name="native_powr")
    ],
    "remainder": [Def("genfloat", ["genfloat", "genfloat"])],
    "remquo": [
        Def("vgenfloat", ["vgenfloat", "vgenfloat", "vint32nptr0"]),
        Def("mgenfloat", ["mgenfloat", "mgenfloat", "intnptr0"], marray_use_loop=True),
        Def("float", ["float", "float", "intptr"]),
        Def("double", ["double", "double", "intptr"]),
        Def("half", ["half", "half", "intptr"]),
    ],
    "rint": [Def("genfloat", ["genfloat"])],
    "rootn": [
        Def("vgenfloat", ["vgenfloat", "vint32n"]),
        Def("mgenfloat", ["mgenfloat", "mintn"], marray_use_loop=True),
        Def("float", ["float", "int"]),
        Def("double", ["double", "int"]),
        Def("half", ["half", "int"]),
    ],
    "round": [Def("genfloat", ["genfloat"])],
    "rsqrt": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_rsqrt")],
    "sin": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_sin")],
    "sincos": [
        Def(
            "vgenfloat",
            ["vgenfloat", "rawvgenfloatptr0"],
            fast_math_custom_invoke=custom_fast_math_sincos_invoke,
        ),
        Def(
            "vgenfloat",
            ["vgenfloat", "ptr0"],
            fast_math_custom_invoke=custom_fast_math_sincos_invoke,
        ),
        Def("mgenfloat", ["mgenfloat", "ptr0"]),
        Def(
            "float",
            ["float", "floatptr"],
            fast_math_custom_invoke=custom_fast_math_sincos_invoke,
        ),
        Def(
            "double",
            ["double", "doubleptr"],
            fast_math_custom_invoke=custom_fast_math_sincos_invoke,
        ),
        Def(
            "half",
            ["half", "halfptr"],
            fast_math_custom_invoke=custom_fast_math_sincos_invoke,
        ),
    ],
    "sinh": [Def("genfloat", ["genfloat"])],
    "sinpi": [Def("genfloat", ["genfloat"])],
    "sqrt": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_sqrt")],
    "tan": [Def("genfloat", ["genfloat"], fast_math_invoke_name="native_tan")],
    "tanh": [Def("genfloat", ["genfloat"])],
    "tanpi": [Def("genfloat", ["genfloat"])],
    "tgamma": [Def("genfloat", ["genfloat"])],
    "trunc": [Def("genfloat", ["genfloat"])],
    # Integer functions
    "abs_diff": [
        Def(
            "sigeninteger",
            ["sigeninteger", "sigeninteger"],
            custom_invoke=get_custom_unsigned_to_signed_scalar_invoke("s_abs_diff"),
            template_scalar_args=True,
        ),
        Def(
            "vigeninteger",
            ["vigeninteger", "vigeninteger"],
            custom_invoke=get_custom_unsigned_to_signed_vec_invoke("s_abs_diff"),
        ),
        Def("migeninteger", ["migeninteger", "migeninteger"], marray_use_loop=True),
        Def(
            "ugeninteger",
            ["ugeninteger", "ugeninteger"],
            invoke_prefix="u_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
    ],
    "add_sat": [
        Def(
            "igeninteger",
            ["igeninteger", "igeninteger"],
            invoke_prefix="s_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def(
            "ugeninteger",
            ["ugeninteger", "ugeninteger"],
            invoke_prefix="u_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
    ],
    "hadd": [
        Def(
            "igeninteger",
            ["igeninteger", "igeninteger"],
            invoke_prefix="s_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def(
            "ugeninteger",
            ["ugeninteger", "ugeninteger"],
            invoke_prefix="u_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
    ],
    "rhadd": [
        Def(
            "igeninteger",
            ["igeninteger", "igeninteger"],
            invoke_prefix="s_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def(
            "ugeninteger",
            ["ugeninteger", "ugeninteger"],
            invoke_prefix="u_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
    ],
    "clz": [
        Def(
            "geninteger",
            ["geninteger"],
            marray_use_loop=True,
            template_scalar_args=True,
        )
    ],
    "ctz": [
        Def(
            "geninteger",
            ["geninteger"],
            marray_use_loop=True,
            template_scalar_args=True,
        )
    ],
    "mad_hi": [
        Def(
            "igeninteger",
            ["igeninteger", "igeninteger", "igeninteger"],
            invoke_prefix="s_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def(
            "ugeninteger",
            ["ugeninteger", "ugeninteger", "ugeninteger"],
            invoke_prefix="u_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
    ],
    "mad_sat": [
        Def(
            "igeninteger",
            ["igeninteger", "igeninteger", "igeninteger"],
            invoke_prefix="s_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def(
            "ugeninteger",
            ["ugeninteger", "ugeninteger", "ugeninteger"],
            invoke_prefix="u_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
    ],
    "mul_hi": [
        Def(
            "igeninteger",
            ["igeninteger", "igeninteger"],
            invoke_prefix="s_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def(
            "ugeninteger",
            ["ugeninteger", "ugeninteger"],
            invoke_prefix="u_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
    ],
    "rotate": [
        Def(
            "geninteger",
            ["geninteger", "geninteger"],
            marray_use_loop=True,
            template_scalar_args=True,
        )
    ],
    "sub_sat": [
        Def(
            "igeninteger",
            ["igeninteger", "igeninteger"],
            invoke_prefix="s_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def(
            "ugeninteger",
            ["ugeninteger", "ugeninteger"],
            invoke_prefix="u_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
    ],
    "upsample": [
        Def(
            "upsampledint0",
            ["int8_t", "uint8_t"],
            invoke_prefix="s_",
            template_scalar_args=True,
        ),
        Def(
            "upsampledint0",
            ["char", "uint8_t"],
            invoke_prefix="s_",
            template_scalar_args=True,
        ),  # TODO: Non-standard. Deprecate.
        Def(
            "upsampledint0",
            ["vint8n", "vuint8n"],
            invoke_prefix="s_",
            template_scalar_args=True,
        ),
        Def("upsampledint0", ["mint8n", "muint8n"]),
        Def(
            "upsampledint0",
            ["uint8_t", "uint8_t"],
            invoke_prefix="u_",
            template_scalar_args=True,
        ),
        Def(
            "upsampledint0",
            ["vuint8n", "vuint8n"],
            invoke_prefix="u_",
            template_scalar_args=True,
        ),
        Def("upsampledint0", ["muint8n", "muint8n"]),
        Def(
            "upsampledint0",
            ["int16_t", "uint16_t"],
            invoke_prefix="s_",
            template_scalar_args=True,
        ),
        Def(
            "upsampledint0",
            ["vint16n", "vuint16n"],
            invoke_prefix="s_",
            template_scalar_args=True,
        ),
        Def("upsampledint0", ["mint16n", "muint16n"]),
        Def(
            "upsampledint0",
            ["uint16_t", "uint16_t"],
            invoke_prefix="u_",
            template_scalar_args=True,
        ),
        Def(
            "upsampledint0",
            ["vuint16n", "vuint16n"],
            invoke_prefix="u_",
            template_scalar_args=True,
        ),
        Def("upsampledint0", ["muint16n", "muint16n"]),
        Def(
            "upsampledint0",
            ["int32_t", "uint32_t"],
            invoke_prefix="s_",
            template_scalar_args=True,
        ),
        Def(
            "upsampledint0",
            ["vint32n", "vuint32n"],
            invoke_prefix="s_",
            template_scalar_args=True,
        ),
        Def("upsampledint0", ["mint32n", "muint32n"]),
        Def(
            "upsampledint0",
            ["uint32_t", "uint32_t"],
            invoke_prefix="u_",
            template_scalar_args=True,
        ),
        Def(
            "upsampledint0",
            ["vuint32n", "vuint32n"],
            invoke_prefix="u_",
            template_scalar_args=True,
        ),
        Def("upsampledint0", ["muint32n", "muint32n"]),
    ],
    "popcount": [
        Def(
            "geninteger",
            ["geninteger"],
            marray_use_loop=True,
            template_scalar_args=True,
        )
    ],
    "mad24": [
        Def(
            "igenint32",
            ["igenint32", "igenint32", "igenint32"],
            invoke_prefix="s_",
            template_scalar_args=True,
        ),
        Def(
            "ugenint32",
            ["ugenint32", "ugenint32", "ugenint32"],
            invoke_prefix="u_",
            template_scalar_args=True,
        ),
    ],
    "mul24": [
        Def(
            "igenint32",
            ["igenint32", "igenint32"],
            invoke_prefix="s_",
            template_scalar_args=True,
        ),
        Def(
            "ugenint32",
            ["ugenint32", "ugenint32"],
            invoke_prefix="u_",
            template_scalar_args=True,
        ),
    ],
    # Common functions
    "clamp": [
        Def(
            "genfloat",
            ["genfloat", "genfloat", "genfloat"],
            invoke_prefix="f",
            template_scalar_args=True,
        ),
        Def(
            "vfloatn",
            ["vfloatn", "float", "float"],
            invoke_prefix="f",
            convert_args=[(1, 0), (2, 0)],
        ),
        Def(
            "vdoublen",
            ["vdoublen", "double", "double"],
            invoke_prefix="f",
            convert_args=[(1, 0), (2, 0)],
        ),
        Def(
            "vhalfn",
            ["vhalfn", "half", "half"],
            invoke_prefix="f",
            convert_args=[(1, 0), (2, 0)],
        ),  # Non-standard. Deprecated.
        Def(
            "igeninteger",
            ["igeninteger", "igeninteger", "igeninteger"],
            invoke_prefix="s_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def(
            "ugeninteger",
            ["ugeninteger", "ugeninteger", "ugeninteger"],
            invoke_prefix="u_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def(
            "vigeninteger",
            ["vigeninteger", "elementtype0", "elementtype0"],
            invoke_prefix="s_",
        ),
        Def(
            "vugeninteger",
            ["vugeninteger", "elementtype0", "elementtype0"],
            invoke_prefix="u_",
        ),
        Def(
            "mgentype",
            ["mgentype", "elementtype0", "elementtype0"],
            marray_use_loop=True,
        ),
    ],
    "degrees": [Def("genfloat", ["genfloat"], template_scalar_args=True)],
    "(max)": [
        Def(
            "genfloat",
            ["genfloat", "genfloat"],
            invoke_name="fmax_common",
            template_scalar_args=True,
        ),
        Def(
            "vfloatn",
            ["vfloatn", "float"],
            invoke_name="fmax_common",
            convert_args=[(1, 0)],
        ),
        Def(
            "vdoublen",
            ["vdoublen", "double"],
            invoke_name="fmax_common",
            convert_args=[(1, 0)],
        ),
        Def(
            "vhalfn",
            ["vhalfn", "half"],
            invoke_name="fmax_common",
            convert_args=[(1, 0)],
        ),  # Non-standard. Deprecated.
        Def(
            "igeninteger",
            ["igeninteger", "igeninteger"],
            invoke_name="s_max",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def(
            "ugeninteger",
            ["ugeninteger", "ugeninteger"],
            invoke_name="u_max",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def("vigeninteger", ["vigeninteger", "elementtype0"], invoke_name="s_max"),
        Def("vugeninteger", ["vugeninteger", "elementtype0"], invoke_name="u_max"),
        Def("mgentype", ["mgentype", "elementtype0"], marray_use_loop=True),
    ],
    "(min)": [
        Def(
            "genfloat",
            ["genfloat", "genfloat"],
            invoke_name="fmin_common",
            template_scalar_args=True,
        ),
        Def(
            "vfloatn",
            ["vfloatn", "float"],
            invoke_name="fmin_common",
            convert_args=[(1, 0)],
        ),
        Def(
            "vdoublen",
            ["vdoublen", "double"],
            invoke_name="fmin_common",
            convert_args=[(1, 0)],
        ),
        Def(
            "vhalfn",
            ["vhalfn", "half"],
            invoke_name="fmin_common",
            convert_args=[(1, 0)],
        ),  # Non-standard. Deprecated.
        Def(
            "igeninteger",
            ["igeninteger", "igeninteger"],
            invoke_name="s_min",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def(
            "ugeninteger",
            ["ugeninteger", "ugeninteger"],
            invoke_name="u_min",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
        Def("vigeninteger", ["vigeninteger", "elementtype0"], invoke_name="s_min"),
        Def("vugeninteger", ["vugeninteger", "elementtype0"], invoke_name="u_min"),
        Def("mgentype", ["mgentype", "elementtype0"], marray_use_loop=True),
    ],
    "mix": [
        Def(
            "genfloat", ["genfloat", "genfloat", "genfloat"], template_scalar_args=True
        ),
        Def("vfloatn", ["vfloatn", "vfloatn", "float"], convert_args=[(2, 0)]),
        Def("vdoublen", ["vdoublen", "vdoublen", "double"], convert_args=[(2, 0)]),
        Def(
            "vhalfn", ["vhalfn", "vhalfn", "half"], convert_args=[(2, 0)]
        ),  # Non-standard. Deprecated.
        Def("mfloatn", ["mfloatn", "mfloatn", "float"]),
        Def("mdoublen", ["mdoublen", "mdoublen", "double"]),
        Def("mhalfn", ["mhalfn", "mhalfn", "half"]),
    ],  # Non-standard. Deprecated.
    "radians": [Def("genfloat", ["genfloat"], template_scalar_args=True)],
    "step": [
        Def("genfloat", ["genfloat", "genfloat"], template_scalar_args=True),
        Def("vfloatn", ["float", "vfloatn"], convert_args=[(0, 1)]),
        Def("vdoublen", ["double", "vdoublen"], convert_args=[(0, 1)]),
        Def(
            "vhalfn", ["half", "vhalfn"], convert_args=[(0, 1)]
        ),  # Non-standard. Deprecated.
        Def("mfloatn", ["float", "mfloatn"]),
        Def("mdoublen", ["double", "mdoublen"]),
        Def("mhalfn", ["half", "mhalfn"]),
    ],  # Non-standard. Deprecated.
    "smoothstep": [
        Def(
            "genfloat", ["genfloat", "genfloat", "genfloat"], template_scalar_args=True
        ),
        Def("vfloatn", ["float", "float", "vfloatn"], convert_args=[(0, 2), (1, 2)]),
        Def(
            "vdoublen", ["double", "double", "vdoublen"], convert_args=[(0, 2), (1, 2)]
        ),
        Def(
            "vhalfn", ["half", "half", "vhalfn"], convert_args=[(0, 2), (1, 2)]
        ),  # Non-standard. Deprecated.
        Def("mfloatn", ["float", "float", "mfloatn"]),
        Def("mdoublen", ["double", "double", "mdoublen"]),
        Def("mhalfn", ["half", "half", "mhalfn"]),
    ],
    "sign": [Def("genfloat", ["genfloat"], template_scalar_args=True)],
    "abs": [
        Def(
            "sigeninteger",
            ["sigeninteger"],
            custom_invoke=get_custom_unsigned_to_signed_scalar_invoke("s_abs"),
            template_scalar_args=True,
        ),
        Def(
            "vigeninteger",
            ["vigeninteger"],
            custom_invoke=get_custom_unsigned_to_signed_vec_invoke("s_abs"),
        ),
        Def("migeninteger", ["migeninteger"], marray_use_loop=True),
        Def(
            "ugeninteger",
            ["ugeninteger"],
            invoke_prefix="u_",
            marray_use_loop=True,
            template_scalar_args=True,
        ),
    ],
    # Geometric functions
    "cross": [
        Def("vfloat3or4", ["vfloat3or4", "vfloat3or4"]),
        Def("vdouble3or4", ["vdouble3or4", "vdouble3or4"]),
        Def("vhalf3or4", ["vhalf3or4", "vhalf3or4"]),
        Def("mfloat3or4", ["mfloat3or4", "mfloat3or4"]),
        Def("mdouble3or4", ["mdouble3or4", "mdouble3or4"]),
        Def("mhalf3or4", ["mhalf3or4", "mhalf3or4"]),
    ],
    "dot": [
        Def("float", ["vgeofloat", "vgeofloat"], invoke_name="Dot"),
        Def("double", ["vgeodouble", "vgeodouble"], invoke_name="Dot"),
        Def("half", ["vgeohalf", "vgeohalf"], invoke_name="Dot"),
        Def("float", ["mgeofloat", "mgeofloat"], invoke_name="Dot"),
        Def("double", ["mgeodouble", "mgeodouble"], invoke_name="Dot"),
        Def("half", ["mgeohalf", "mgeohalf"], invoke_name="Dot"),
        Def(
            "sgenfloat",
            ["sgenfloat", "sgenfloat"],
            template_scalar_args=True,
            custom_invoke=(
                lambda return_types, arg_types, arg_names: "  return "
                + " * ".join(arg_names)
                + ";"
            ),
        ),
    ],
    "distance": [
        Def("float", ["gengeofloat", "gengeofloat"], template_scalar_args=True),
        Def("double", ["gengeodouble", "gengeodouble"], template_scalar_args=True),
        Def("half", ["gengeohalf", "gengeohalf"], template_scalar_args=True),
    ],
    "length": [
        Def("float", ["gengeofloat"], template_scalar_args=True),
        Def("double", ["gengeodouble"], template_scalar_args=True),
        Def("half", ["gengeohalf"], template_scalar_args=True),
    ],
    "normalize": [
        Def("gengeofloat", ["gengeofloat"], template_scalar_args=True),
        Def("gengeodouble", ["gengeodouble"], template_scalar_args=True),
        Def("gengeohalf", ["gengeohalf"], template_scalar_args=True),
    ],
    "fast_distance": [
        Def("float", ["gengeofloat", "gengeofloat"], template_scalar_args=True)
    ],
    "fast_length": [Def("float", ["gengeofloat"], template_scalar_args=True)],
    "fast_normalize": [Def("gengeofloat", ["gengeofloat"], template_scalar_args=True)],
    # Relational functions
    "isequal": [
        RelDef(
            "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdEqual"
        ),
        RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdEqual"),
        RelDef("boolelements0", ["mgenfloat", "mgenfloat"]),
    ],
    "isnotequal": [
        RelDef(
            "samesizesignedint0",
            ["vgenfloat", "vgenfloat"],
            invoke_name="FUnordNotEqual",
        ),
        RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FUnordNotEqual"),
        RelDef("boolelements0", ["mgenfloat", "mgenfloat"]),
    ],
    "isgreater": [
        RelDef(
            "samesizesignedint0",
            ["vgenfloat", "vgenfloat"],
            invoke_name="FOrdGreaterThan",
        ),
        RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdGreaterThan"),
        RelDef("boolelements0", ["mgenfloat", "mgenfloat"]),
    ],
    "isgreaterequal": [
        RelDef(
            "samesizesignedint0",
            ["vgenfloat", "vgenfloat"],
            invoke_name="FOrdGreaterThanEqual",
        ),
        RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdGreaterThanEqual"),
        RelDef("boolelements0", ["mgenfloat", "mgenfloat"]),
    ],
    "isless": [
        RelDef(
            "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdLessThan"
        ),
        RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdLessThan"),
        RelDef("boolelements0", ["mgenfloat", "mgenfloat"]),
    ],
    "islessequal": [
        RelDef(
            "samesizesignedint0",
            ["vgenfloat", "vgenfloat"],
            invoke_name="FOrdLessThanEqual",
        ),
        RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdLessThanEqual"),
        RelDef("boolelements0", ["mgenfloat", "mgenfloat"]),
    ],
    "islessgreater": [
        RelDef(
            "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="FOrdNotEqual"
        ),
        RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="FOrdNotEqual"),
        RelDef("boolelements0", ["mgenfloat", "mgenfloat"]),
    ],
    "isfinite": [
        RelDef("samesizesignedint0", ["vgenfloat"], invoke_name="IsFinite"),
        RelDef("bool", ["sgenfloat"], invoke_name="IsFinite"),
        RelDef("boolelements0", ["mgenfloat"]),
    ],
    "isinf": [
        RelDef("samesizesignedint0", ["vgenfloat"], invoke_name="IsInf"),
        RelDef("bool", ["sgenfloat"], invoke_name="IsInf"),
        RelDef("boolelements0", ["mgenfloat"]),
    ],
    "isnan": [
        RelDef("samesizesignedint0", ["vgenfloat"], invoke_name="IsNan"),
        RelDef("bool", ["sgenfloat"], invoke_name="IsNan"),
        RelDef("boolelements0", ["mgenfloat"]),
    ],
    "isnormal": [
        RelDef("samesizesignedint0", ["vgenfloat"], invoke_name="IsNormal"),
        RelDef("bool", ["sgenfloat"], invoke_name="IsNormal"),
        RelDef("boolelements0", ["mgenfloat"]),
    ],
    "isordered": [
        RelDef("samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="Ordered"),
        RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="Ordered"),
        RelDef("boolelements0", ["mgenfloat", "mgenfloat"]),
    ],
    "isunordered": [
        RelDef(
            "samesizesignedint0", ["vgenfloat", "vgenfloat"], invoke_name="Unordered"
        ),
        RelDef("bool", ["sgenfloat", "sgenfloat"], invoke_name="Unordered"),
        RelDef("boolelements0", ["mgenfloat", "mgenfloat"]),
    ],
    "signbit": [
        RelDef("samesizesignedint0", ["vgenfloat"], invoke_name="SignBitSet"),
        RelDef("bool", ["sgenfloat"], invoke_name="SignBitSet"),
        RelDef("boolelements0", ["mgenfloat"]),
    ],
    "any": [
        Def(
            "int", ["vigeninteger"], custom_invoke=get_custom_any_all_vec_invoke("Any")
        ),
        Def(
            "bool",
            ["sigeninteger"],
            template_scalar_args=True,
            custom_invoke=(
                lambda return_type, arg_types, arg_names: f"  return bool(int(detail::msbIsSet({arg_names[0]})));"
            ),
        ),
        Def(
            "bool",
            ["migeninteger"],
            custom_invoke=get_custom_any_all_marray_invoke("any"),
        ),
    ],
    "all": [
        Def(
            "int", ["vigeninteger"], custom_invoke=get_custom_any_all_vec_invoke("All")
        ),
        Def(
            "bool",
            ["sigeninteger"],
            template_scalar_args=True,
            custom_invoke=(
                lambda return_type, arg_types, arg_names: f"  return bool(int(detail::msbIsSet({arg_names[0]})));"
            ),
        ),
        Def(
            "bool",
            ["migeninteger"],
            custom_invoke=get_custom_any_all_marray_invoke("all"),
        ),
    ],
    "bitselect": [
        Def("vgentype", ["vgentype", "vgentype", "vgentype"]),
        Def(
            "sgentype", ["sgentype", "sgentype", "sgentype"], template_scalar_args=True
        ),
        Def("mgentype", ["mgentype", "mgentype", "mgentype"], marray_use_loop=True),
    ],
    "select": [
        Def("vint8n", ["vint8n", "vint8n", "vint8n"]),
        Def("vint16n", ["vint16n", "vint16n", "vint16n"]),
        Def("vint32n", ["vint32n", "vint32n", "vint32n"]),
        Def("vint64n", ["vint64n", "vint64n", "vint64n"]),
        Def("vuint8n", ["vuint8n", "vuint8n", "vint8n"]),
        Def("vuint16n", ["vuint16n", "vuint16n", "vint16n"]),
        Def("vuint32n", ["vuint32n", "vuint32n", "vint32n"]),
        Def("vuint64n", ["vuint64n", "vuint64n", "vint64n"]),
        Def("vfloatn", ["vfloatn", "vfloatn", "vint32n"]),
        Def("vdoublen", ["vdoublen", "vdoublen", "vint64n"]),
        Def("vhalfn", ["vhalfn", "vhalfn", "vint16n"]),
        Def("vint8n", ["vint8n", "vint8n", "vuint8n"]),
        Def("vint16n", ["vint16n", "vint16n", "vuint16n"]),
        Def("vint32n", ["vint32n", "vint32n", "vuint32n"]),
        Def("vint64n", ["vint64n", "vint64n", "vuint64n"]),
        Def("vuint8n", ["vuint8n", "vuint8n", "vuint8n"]),
        Def("vuint16n", ["vuint16n", "vuint16n", "vuint16n"]),
        Def("vuint32n", ["vuint32n", "vuint32n", "vuint32n"]),
        Def("vuint64n", ["vuint64n", "vuint64n", "vuint64n"]),
        Def("vfloatn", ["vfloatn", "vfloatn", "vuint32n"]),
        Def("vdoublen", ["vdoublen", "vdoublen", "vuint64n"]),
        Def("vhalfn", ["vhalfn", "vhalfn", "vuint16n"]),
        Def(
            "sgentype",
            ["sgentype", "sgentype", "bool"],
            template_scalar_args=True,
            custom_invoke=custom_bool_select_invoke,
        ),
        Def("mgentype", ["mgentype", "mgentype", "mbooln"], marray_use_loop=True),
    ],
}
# List of all builtins definitions in the sycl::native namespace.
native_builtins = {
    "cos": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
    "divide": [Def("genfloatf", ["genfloatf", "genfloatf"], invoke_prefix="native_")],
    "exp": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
    "exp2": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
    "exp10": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
    "log": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
    "log2": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
    "log10": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
    "powr": [Def("genfloatf", ["genfloatf", "genfloatf"], invoke_prefix="native_")],
    "recip": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
    "rsqrt": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
    "sin": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
    "sqrt": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
    "tan": [Def("genfloatf", ["genfloatf"], invoke_prefix="native_")],
}
# List of all builtins definitions in the sycl::half_precision namespace.
half_precision_builtins = {
    "cos": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
    "divide": [Def("genfloatf", ["genfloatf", "genfloatf"], invoke_prefix="half_")],
    "exp": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
    "exp2": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
    "exp10": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
    "log": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
    "log2": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
    "log10": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
    "powr": [Def("genfloatf", ["genfloatf", "genfloatf"], invoke_prefix="half_")],
    "recip": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
    "rsqrt": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
    "sin": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
    "sqrt": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
    "tan": [Def("genfloatf", ["genfloatf"], invoke_prefix="half_")],
}

# All builtin definitions grouped by their namespace.
builtins_groups = [
    (None, sycl_builtins),
    ("native", native_builtins),
    ("half_precision", half_precision_builtins),
]

### GENERATION


def select_from_mapping(mappings, arg_types, arg_type):
    """
    Selects a type from a mapping. If the mapped type needs a parent-type, they
    are instantiated with it here.
    """
    mapping = mappings[arg_type]
    # In some cases we may need to limit definitions to smaller than geninteger so
    # check for the different possible ones.
    if isinstance(mapping, ElementType):
        parent_mapping = mappings[arg_types[mapping.parent_idx]]
        return InstantiatedElementType(parent_mapping, mapping.parent_idx)
    if isinstance(mapping, ConversionTraitType):
        parent_mapping = mappings[arg_types[mapping.parent_idx]]
        return InstantiatedConversionTraitType(
            parent_mapping, mapping.trait, mapping.parent_idx
        )
    return mapping


def instantiate_arg(idx, arg):
    """Instantiates an argument by its type."""
    if isinstance(arg, TemplatedType):
        # Instantiate the template type by giving it a name based on its index.
        return InstantiatedTemplatedArg(f"T{idx}", arg)
    if isinstance(arg, RawPtr):
        # Instantiate the pointed-to type.
        return RawPtr(instantiate_arg(idx, arg.element_type))
    if isinstance(arg, InstantiatedElementType):
        # Instantiate the referenced type.
        return InstantiatedElementType(
            instantiate_arg(arg.parent_idx, arg.referenced_type), arg.parent_idx
        )
    if isinstance(arg, InstantiatedConversionTraitType):
        # Instantiate the parent type.
        return InstantiatedConversionTraitType(
            instantiate_arg(arg.parent_idx, arg.parent_type), arg.trait, arg.parent_idx
        )
    return arg


def instantiate_return_type(return_type, instantiated_args):
    """Instantiates a return type by its type."""
    if isinstance(return_type, TemplatedType):
        # Instantiate the template type by giving it the first template argument
        # in the function.
        first_templated = find_first_template_arg(instantiated_args)
        return InstantiatedTemplatedReturnType(str(first_templated), return_type)
    if isinstance(return_type, RawPtr):
        # Instantiate the pointed-to type.
        return RawPtr(
            instantiate_return_type(return_type.element_type, instantiated_args)
        )
    if isinstance(return_type, InstantiatedElementType):
        # Instantiate the referenced type.
        return InstantiatedElementType(
            instantiate_return_type(return_type.referenced_type, instantiated_args),
            return_type.parent_idx,
        )
    if isinstance(return_type, InstantiatedConversionTraitType):
        # Instantiate the parent type.
        return InstantiatedConversionTraitType(
            instantiate_return_type(return_type.parent_type, instantiated_args),
            return_type.trait,
            return_type.parent_idx,
        )
    return return_type


def is_valid_combination(return_type, arg_types):
    """Checks that the selected types form a valid combination."""
    marray_arg_seen = False
    vec_arg_seen = False
    raw_ptr_seen = False
    for t in [return_type] + arg_types:
        if is_marray_arg(t):
            marray_arg_seen = True
        elif is_vec_arg(t):
            vec_arg_seen = True
        elif isinstance(t, RawPtr):
            raw_ptr_seen = True
    # Following rules apply:
    # 1. vec and marray cannot be in the same combination.
    # 2. marray and raw pointers were never supported together and shouldn't be.
    return not (marray_arg_seen and (vec_arg_seen or raw_ptr_seen))


def convert_scalars_to_templated(type_list):
    """
    Converts the scalar arguments in the type list to a single templated scalar
    type.
    """
    result = []
    scalars = []
    for t in type_list:
        if isinstance(t, str):
            scalars.append(t)
        else:
            result.append(t)
    if len(scalars) > 0:
        result.append(TemplatedScalar(scalars))
    return result


def type_combinations(return_type, arg_types, template_scalars):
    """
    Generates all return and argument type combinations for a given builtin
    definition.
    """
    unique_types = list(dict.fromkeys(arg_types))
    unique_type_lists = [builtin_types[unique_type] for unique_type in unique_types]
    if template_scalars:
        unique_type_lists = [
            convert_scalars_to_templated(type_list) for type_list in unique_type_lists
        ]
    if return_type not in unique_types:
        # Add return type after scalars have been turned to template arguments if
        # it is unique, to avoid undeducible return types.
        unique_types.append(return_type)
        unique_type_lists.append(builtin_types[return_type])
    combinations = list(itertools.product(*unique_type_lists))
    result = []
    for combination in combinations:
        mappings = dict(zip(unique_types, combination))
        mapped_return_type = select_from_mapping(mappings, arg_types, return_type)
        mapped_arg_types = [
            select_from_mapping(mappings, arg_types, arg_type) for arg_type in arg_types
        ]
        instantiated_arg_types = [
            instantiate_arg(idx, arg_type)
            for idx, arg_type in enumerate(mapped_arg_types)
        ]
        instantiated_return_type = instantiate_return_type(
            mapped_return_type, instantiated_arg_types
        )
        if is_valid_combination(instantiated_return_type, instantiated_arg_types):
            result.append((instantiated_return_type, instantiated_arg_types))
    return result


def get_all_template_args(arg_types):
    """Returns all the unique template arguments in the argument type list."""
    result = []
    for arg_type in arg_types:
        if isinstance(arg_type, InstantiatedTemplatedArg):
            result.append(arg_type)
        if isinstance(arg_type, RawPtr):
            result += get_all_template_args([arg_type.element_type])
    return result


def get_arg_requirement(arg_type, arg_type_name):
    """Returns the conjunction of all requirements for the given argument."""
    return "(" + (" && ".join(arg_type.get_requirements(arg_type_name))) + ")"


def get_all_same_type_requirement(template_names):
    """
    Returns the template requirement that all the template type names in the
    given list have the same type.
    """
    template_name_args = ", ".join(template_names)
    return f"detail::check_all_same_op_type_v<{template_name_args}>"


def get_func_return(return_type, arg_types):
    """
    Generates the function return type, including SFINAE if the generated builtin
    has template arguments.
    """
    temp_args = get_all_template_args(arg_types)
    if len(temp_args) > 0:
        type_groups = defaultdict(lambda: [])
        for temp_arg in temp_args:
            type_groups[temp_arg.templated_type].append(temp_arg.template_name)
        requirements = []
        for temp_type, temp_names in type_groups.items():
            # Add the requirements of one of the template arguments in the same type
            # group. Since they are required to have the same type it is enough to
            # just check one.
            requirements.append(get_arg_requirement(temp_type, temp_names[0]))
            # Add a requirement that all template arguments of the same type group
            # have the same type. No point in doing so is if is along in the group.
            if len(temp_names) > 1:
                requirements.append(get_all_same_type_requirement(temp_names))
        conjunc_reqs = " && ".join(requirements)
        return f"std::enable_if_t<{conjunc_reqs}, {return_type}>"
    return str(return_type)


def get_template_args(arg_types):
    """
    Gets a list of the template arguments for the given argument types. If there
    are multi_ptr arguments the space and decoration is added to the end of the
    template arguments.
    """
    temp_args = get_all_template_args(arg_types)
    return [f"typename {temp_arg.template_name}" for temp_arg in temp_args]


def get_deprecation(builtin, return_type, arg_types):
    """Gets the deprecation statement for a given builtin."""
    if builtin.deprecation_message:
        return f'__SYCL_DEPRECATED("{builtin.deprecation_message}")\n'
    for t in [return_type] + arg_types:
        if hasattr(t, "deprecation_message") and t.deprecation_message:
            return f'__SYCL_DEPRECATED("{t.deprecation_message}")\n'
    return ""


def get_func_prefix(builtin, return_type, arg_types):
    """
    Get the prefix for a builtin function definitions. This can include `inline`,
    the template definition, and any associated deprecation attribute.
    """
    template_args = get_template_args(arg_types)
    func_deprecation = get_deprecation(builtin, return_type, arg_types)
    result = ""
    if template_args:
        result += "template <%s>\n" % (", ".join(template_args))
    if func_deprecation:
        result += func_deprecation
    if not template_args:
        result += "inline "
    return result


def generate_builtin(builtin_name, namespace, builtin, return_type, arg_types):
    """Generates a builtin function definition."""
    func_prefix = get_func_prefix(builtin, return_type, arg_types)
    func_return = get_func_return(return_type, arg_types)
    arg_names = ["a%i" % i for i in range(len(arg_types))]
    func_args = ", ".join(["%s %s" % arg for arg in zip(arg_types, arg_names)])
    invoke = builtin.get_invoke(
        builtin_name, namespace, return_type, arg_types, arg_names
    )
    return f"""
{func_prefix}{func_return} {builtin_name}({func_args}) __NOEXC {{
{invoke}
}}
"""


def generate_builtins(builtins, namespace):
    """
    Generates all builtins for the builtin definitions specified. Returns the
    resulting builtin function definitions in three different lists: scalar,
    vector and marray builtins.
    """
    scalar_result = []
    vector_result = []
    marray_result = []
    for builtin_name, builtin_defs in builtins.items():
        for builtin_def in builtin_defs:
            combs = type_combinations(
                builtin_def.return_type,
                builtin_def.arg_types,
                builtin_def.template_scalar_args,
            )
            for return_t, arg_ts in combs:
                generated_builtin = generate_builtin(
                    builtin_name, namespace, builtin_def, return_t, arg_ts
                )
                if any([is_marray_arg(arg_t) for arg_t in arg_ts]):
                    marray_result.append(generated_builtin)
                elif any([is_vec_arg(arg_t) for arg_t in arg_ts]):
                    vector_result.append(generated_builtin)
                else:
                    scalar_result.append(generated_builtin)
    return (scalar_result, vector_result, marray_result)


def generate_file(directory, file_name, includes, generated_builtins):
    """Generates a builtins header."""
    instantiated_includes = "\n".join([f"#include <{inc}>" for inc in includes])

    if file_name == "builtins_scalar_gen.hpp":
        include = "sycl/builtins_utils_scalar.hpp"
    else:
        include = "sycl/builtins_utils_vec.hpp"

    with open(os.path.join(directory, file_name), "w+") as f:
        f.write(
            f"""
//==--------- {file_name} - SYCL generated built-in functions ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NOTE: This file was generated and should not be changed!

#pragma once

{instantiated_includes}

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

namespace sycl {{
inline namespace _V1 {{
"""
        )

        for namespace, builtins in generated_builtins:
            if namespace:
                f.write(f"\nnamespace {namespace} {{")
            f.write("".join(builtins))
            if namespace:
                f.write(f"}} // namespace {namespace}\n")

        f.write(
            """
} // namespace _V1
} // namespace sycl

#undef __NOEXC
"""
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Invalid number of arguments! Must be given an output path.")

    # Generate the builtins.
    scalar_builtins = []
    vector_builtins = []
    marray_builtins = []
    for namespace, builtins in builtins_groups:
        (sb, vb, mb) = generate_builtins(builtins, namespace)
        scalar_builtins.append((namespace, sb))
        vector_builtins.append((namespace, vb))
        marray_builtins.append((namespace, mb))

    # Write the builtins to new header files, separated by whether or not they
    # are scalar, vector or marray builtins.
    file_path = sys.argv[1]
    generate_file(
        file_path,
        "builtins_scalar_gen.hpp",
        ["sycl/builtins_utils_scalar.hpp"],
        scalar_builtins,
    )
    generate_file(
        file_path,
        "builtins_vector_gen.hpp",
        ["sycl/builtins_utils_vec.hpp"],
        vector_builtins,
    )
    generate_file(
        file_path,
        "builtins_marray_gen.hpp",
        ["sycl/builtins_scalar_gen.hpp", "sycl/builtins_vector_gen.hpp"],
        marray_builtins,
    )
