================================
SPIR-V representation in LLVM IR
================================
.. contents::
   :local:

Overview
========

As one of the goals of SPIR-V is to `"map easily to other IRs, including LLVM
IR" <https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_goals>`_,
most of SPIR-V entities (global variables, constants, types, functions, basic
blocks, instructions) have straightforward counterparts in LLVM. Therefore the
focus of this document is those entities in SPIR-V which do not map to LLVM in
an obvious way. These include:

 * SPIR-V types mapped to LLVM types
 * SPIR-V instructions mapped to LLVM function calls
 * SPIR-V extended instructions mapped to LLVM function calls
 * SPIR-V builtin variables mapped to LLVM function calls or LLVM global variables
 * SPIR-V instructions mapped to LLVM metadata
 * SPIR-V types mapped to LLVM opaque types
 * SPIR-V decorations mapped to LLVM metadata or named attributes
 * Additional requirements for LLVM module

SPIR-V Types Mapped to LLVM Types
=================================
Limited to this section, we define the following common postfix.

* {Access} - Postifix indicating the access qualifier.
{Access} take integer literal values which are defined by the SPIR-V spec.

OpTypeImage
-----------
OpTypeImage is mapped to LLVM opaque type
spirv.Image._{SampledType}_{Dim}_{Depth}_{Arrayed}_{MS}_{Sampled}_{Format}_{Access}
and mangled as __spirv_Image__{SampledType}_{Dim}_{Depth}_{Arrayed}_{MS}_{Sampled}_{Format}_{Access},

where

* {SampledType}={float|half|int|uint|void} - Postfix indicating the sampled data type
  - void for unknown sampled data type
* {Dim} - Postfix indicating the dimension of the image
* {Depth} - Postfix indicating whether the image is a depth image
* {Arrayed} - Postfix indicating whether the image is arrayed image
* {MS} - Postfix indicating whether the image is multi-sampled
* {Sampled} - Postfix indicating whether the image is associated with sampler
* {Format} - Postfix indicating the image format

Postfixes {Dim}, {Depth}, {Arrayed}, {MS}, {Sampled} and {Format} take integer
literal values which are defined by the SPIR-V spec.

OpTypeSampledImage
------------------
OpTypeSampledImage is mapped to LLVM opaque type
spirv.SampledImage._{Postfixes} and mangled as __spirv_SampledImage__{Postfixes},
where {Postfixes} are the same as the postfixes of the original image type, as
defined above in this section.

OpTypePipe
----------
OpTypePipe is mapped to LLVM opaque type
spirv.Pipe._{Access} and mangled as __spirv_Pipe__{Access}.

Other SPIR-V Types
------------------
* OpTypeEvent
* OpTypeDeviceEvent
* OpTypeReserveId
* OpTypeQueue
* OpTypeSampler
* OpTypePipeStorage (SPIR-V 1.1)
The above SPIR-V types are mapped to LLVM opaque type spirv.{TypeName} and
mangled as __spirv_{TypeName}, where {TypeName} is the name of the SPIR-V
type with "OpType" removed, e.g., OpTypeEvent is mapped to spirv.Event and
mangled as __spirv_Event.

Address spaces
--------------

The following
`SPIR-V storage classes <https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#Storage_Class>`_
are naturally represented as LLVM IR address spaces with the following mapping:

====================    ====================================
SPIR-V storage class    LLVM IR address space
====================    ====================================
``Function``            No address space or ``addrspace(0)``
``CrossWorkgroup``      ``addrspace(1)``
``UniformConstant``     ``addrspace(2)``
``Workgroup``           ``addrspace(3)``
``Generic``             ``addrspace(4)``
====================    ====================================

SPIR-V extensions are allowed to add new storage classes. For example,
SPV_INTEL_usm_storage_classes extension adds ``DeviceOnlyINTEL`` and
``HostOnlyINTEL`` storage classes which are mapped to ``addrspace(5)`` and
``addrspace(6)`` respectively.

SPIR-V Instructions Mapped to LLVM Function Calls
=================================================

Some SPIR-V instructions which can be included in basic blocks do not have
corresponding LLVM instructions or intrinsics. These SPIR-V instructions are
represented by function calls in LLVM. The function corresponding to a SPIR-V
instruction is termed SPIR-V builtin function and its name is `IA64 mangled
<https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling>`_ with extensions
for SPIR-V specific types. The unmangled name of a SPIR-V builtin function
follows the convention

.. code-block:: c

  __spirv_{OpCodeName}{_OptionalPostfixes}

where {OpCodeName} is the op code name of the SPIR-V instructions without the
"Op" prefix, e.g. EnqueueKernel. {OptionalPostfixes} are optional postfixes to
specify decorations for the SPIR-V instruction. The SPIR-V op code name and
each postfix does not contain "_".

SPIR-V builtin functions accepts all argument types accepted by the
corresponding SPIR-V instructions. The literal operands of extended
instruction are mapped to function call arguments with type i32.

Optional Postfixes for SPIR-V Builtin Function Names
----------------------------------------------------

SPIR-V builtin functions corresponding to the following SPIR-V instructions are
postfixed following the order specified as below:

 * Instructions having identical argument types but different return types are postfixed with "_R{ReturnType}" where
    - {ReturnType} = {ScalarType}|{VectorType}
    - {ScalarType} = char|uchar|short|ushort|int|uint|long|ulong|half|float|double|bool
    - {VectorType} = {ScalarType}{2|3|4|8|16}
 * Instructions with saturation decoration are postfixed with "_sat"
 * Instructions with floating point rounding mode decoration are postfixed with "_rtp|_rtn|_rtz|_rte"

SPIR-V Builtin Conversion Function Names
----------------------------------------

The unmangled names of SPIR-V builtin conversion functions follow the convention:

.. code-block:: c

  __spirv_{ConversionOpCodeName}_R{ReturnType}{_sat}{_rtp|_rtn|_rtz|_rte}

where

 * {ConversionOpCodeName} = ConvertFToU|ConvertFToS|ConvertUToF|ConvertUToS|UConvert|SConvert|FConvert|SatConvertSToU|SatConvertUToS

SPIR-V Builtin Reinterpret / Bitcast Function Names
---------------------------------------------------

The unmangled names of SPIR-V builtin reinterpret / bitcast functions follow the convention:

.. code-block:: c

  __spirv_{BitcastOpCodeName}_R{ReturnType}

SPIR-V Builtin ImageSample Function Names
----------------------------------------

The unmangled names of SPIR-V builtin ImageSample functions follow the convention:

.. code-block:: c

  __spirv_{ImageSampleOpCodeName}_R{ReturnType}

SPIR-V Builtin GenericCastToPtr Function Name
----------------------------------------

The unmangled names of SPIR-V builtin GenericCastToPtrExplicit function follow the convention:

.. code-block:: c

  __spirv_GenericCastToPtrExplicit_To{Global|Local|Private}

SPIR-V Builtin BuildNDRange Function Name
----------------------------------------

The unmangled names of SPIR-V builtin BuildNDRange functions follow the convention:

.. code-block:: c

  __spirv_{BuildNDRange}_{1|2|3}D

SPIR-V 1.1 Builtin CreatePipeFromPipeStorage Function Name
----------------------------------------

The unmangled names of SPIR-V builtin CreatePipeFromPipeStorage function follow the convention:

.. code-block:: c

  __spirv_CreatePipeFromPipeStorage_{read|write}

SPIR-V Extended Instructions Mapped to LLVM Function Calls
==========================================================

SPIR-V extended instructions are mapped to LLVM function calls. The function
name is IA64 mangled and the unmangled name has the format

.. code-block:: c

  __spirv_{ExtendedInstructionSetName}_{ExtendedInstrutionName}{__OptionalPostfixes}

where {ExtendedInstructionSetName} for OpenCL is "ocl".

The translated functions accepts all argument types accepted by the
corresponding SPIR-V instructions. The literal operands of extended
instruction are mapped to function call arguments with type i32.

The optional postfixes take the same format as SPIR-V builtin functions. The first postfix
starts with two underscores to facilitate identification since extended instruction name
may contain underscore. The remaining postfixes start with one underscore.

OpenCL Extended Builtin Vector Load Function Names
----------------------------------------

The unmangled names of OpenCL extended vector load functions follow the convention:

.. code-block:: c

  __spirv_ocl_{VectorLoadOpCodeName}__R{ReturnType}

where

 * {VectorLoadOpCodeName} = vloadn|vload_half|vload_halfn|vloada_halfn


SPIR-V Builtin Variables Mapped to LLVM Function Calls or LLVM Global Variables
===============================================================================

By default each access of SPIR-V builtin variable's value is mapped to LLVM
function call. The unmangled names of these functions follow the convention:

.. code-block:: c

  __spirv_BuiltIn{VariableName}

In case if SPIR-V builtin variable has vector type, the corresponding
LLVM function will have an integer argument, so each access of the variable's
scalar component is mapped to a function call with index argument, i.e.:

.. code-block:: llvm

  ; For scalar variables
  ; SPIR-V
  OpDecorate %__spirv_BuiltInGlobalInvocationId BuiltIn GlobalInvocationId
  %13 = OpLoad %uint %__spirv_BuiltInGlobalLinearId Aligned 4

  ; Will be transformed into the following LLVM IR:
  %0 = call spir_func i32 @_Z29__spirv_BuiltInGlobalLinearIdv()

  ; For vector variables
  ; SPIRV
  OpDecorate %__spirv_BuiltInGlobalInvocationId BuiltIn GlobalInvocationId
  %14 = OpLoad %v3ulong %__spirv_BuiltInGlobalInvocationId Aligned 32
  %15 = OpCompositeExtract %ulong %14 1

  ; Can be transformed into the following LLVM IR:
  %0 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1)

  ; However SPIRV-LLVM translator will transform it to the following pattern:
  %1 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0)
  %2 = insertelement <3 x i64> undef, i64 %1, i32 0
  %3 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1)
  %4 = insertelement <3 x i64> %2, i64 %3, i32 1
  %5 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 2)
  %6 = insertelement <3 x i64> %4, i64 %5, i32 2
  %7 = extractelement <3 x i64> %6, i32 1
  ; In case some actions are performed with the variable's value in vector form.

SPIR-V builtin variables can also be mapped to LLVM global variables with
unmangled name __spirv_BuiltIn{Name}.

The representation with variables is closer to SPIR-V, so it is easier to
translate from SPIR-V to LLVM and back using it.
Hovewer in languages like OpenCL the functionality covered by SPIR-V builtin
variables is usually represented by builtin functions, so it is easier to
translate from/to SPIR-V friendly IR to/from LLVM IR produced from OpenCL-like
source languages. That is why both forms of mapping are supported.

SPIR-V instructions mapped to LLVM metadata
===========================================

SPIR-V specification allows multiple module scope instructions, whereas LLVM
named metadata must be unique, so encoding of such instructions has the
following format:

.. code-block:: llvm

  !spirv.<OpCodeName> = !{!<InstructionMetadata1>, !<InstructionMetadata2>, ..}
  !<InstructionMetadata1> = !{<Operand1>, <Operand2>, ..}
  !<InstructionMetadata2> = !{<Operand1>, <Operand2>, ..}

+--------------------+---------------------------------------------------------+
| SPIR-V instruction | LLVM IR                                                 |
+====================+=========================================================+
| OpSource           | .. code-block:: llvm                                    |
|                    |                                                         |
|                    |    !spirv.Source = !{!0}                                |
|                    |    !0 = !{i32 3, i32 66048, !1}                         |
|                    |    ; 3 - OpenCL_C                                       |
|                    |    ; 66048 = 0x10200 - OpenCL version 1.2               |
|                    |    ; !1 - optional file id.                             |
|                    |    !1 = !{!"/tmp/opencl/program.cl"}                    |
+--------------------+---------------------------------------------------------+
| OpSourceExtension  | .. code-block:: llvm                                    |
|                    |                                                         |
|                    |    !spirv.SourceExtension = !{!0, !1}                   |
|                    |    !0 = !{!"cl_khr_fp16"}                               |
|                    |    !1 = !{!"cl_khr_gl_sharing"}                         |
+--------------------+---------------------------------------------------------+
| OpExtension        | .. code-block:: llvm                                    |
|                    |                                                         |
|                    |    !spirv.Extension = !{!0}                             |
|                    |    !0 = !{!"SPV_KHR_expect_assume"}                     |
+--------------------+---------------------------------------------------------+
| OpCapability       | .. code-block:: llvm                                    |
|                    |                                                         |
|                    |    !spirv.Capability = !{!0}                            |
|                    |    !0 = !{i32 10} ; Float64 - program uses doubles      |
+--------------------+---------------------------------------------------------+
| OpExecutionMode    | .. code-block:: llvm                                    |
|                    |                                                         |
|                    |    !spirv.ExecutionMode = !{!0}                         |
|                    |    !0 = !{void ()* @worker, i32 30, i32 262149}         |
|                    |    ; Set execution mode with id 30 (VecTypeHint) and    |
|                    |    ; literal `262149` operand.                          |
+--------------------+---------------------------------------------------------+
| Generator's magic  | .. code-block:: llvm                                    |
| number - word # 2  |                                                         |
| in SPIR-V module   |    !spirv.Generator = !{!0}                             |
|                    |    !0 = !{i16 6, i16 123}                               |
|                    |    ; 6 - Generator Id, 123 - Generator Version          |
+--------------------+---------------------------------------------------------+

For example:

.. code-block:: llvm

  !spirv.Source = !{!0}
  !spirv.SourceExtension = !{!2, !3}
  !spirv.Extension = !{!2}
  !spirv.Capability = !{!4}
  !spirv.MemoryModel = !{!5}
  !spirv.EntryPoint = !{!6 ,!7}
  !spirv.ExecutionMode = !{!8, !9}
  !spirv.Generator = !{!10 }

  ; 3 - OpenCL_C, 102000 - OpenCL version 1.2, !1 - optional file id.
  !0 = !{i32 3, i32 102000, !1}
  !1 = !{!"/tmp/opencl/program.cl"}
  !2 = !{!"cl_khr_fp16"}
  !3 = !{!"cl_khr_gl_sharing"}
  !4 = !{i32 10}                ; Float64 - program uses doubles
  !5 = !{i32 1, i32 2}     ; 1 - 32-bit addressing model, 2 - OpenCL memory model
  !6 = !{i32 6, TBD, !"kernel1", TBD}
  !7 = !{i32 6, TBD, !"kernel2", TBD}
  !8 = !{!6, i32 18, i32 16, i32 1, i32 1}     ; local size hint <16, 1, 1> for 'kernel1'
  !9 = !{!7, i32 32}     ; independent forward progress is required for 'kernel2'
  !10 = !{i16 6, i16 123} ; 6 - Generator Id, 123 - Generator Version 

Additional requirements for LLVM module
=======================================

Target triple and datalayout string
-----------------------------------

Target triple architecture must be ``spir`` (32-bit architecture) or ``spir64``
(64-bit architecture) and ``datalayout`` string must be aligned with OpenCL
environment specification requirements for data type sizes and alignments (e.g.
3-element vector must have 4-element vector alignment). For example:

.. code-block:: llvm

   target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
   target triple = "spir-unknown-unknown"

Target triple architecture is translated to
`addressing model operand <https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_addressing_model_a_addressing_model>`_
of
`OpMemoryModel <https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_mode_setting_a_mode_setting_instructions>`_
SPIR-V instruction.

- ``spir`` -> Physical32
- ``spir64`` -> Physical64

Calling convention
------------------

``OpEntryPoint`` information is represented in LLVM IR in calling convention.
A function with ``spir_kernel`` calling convention will be translated as an entry
point of the SPIR-V module.

Function metadata
-----------------

Some kernel parameter information is stored in LLVM IR as a function metadata.

For example:

.. code-block:: llvm

  !kernel_arg_addr_space !1
  !kernel_arg_access_qual !2
  !kernel_arg_type !3
  !kernel_arg_base_type !4
  !kernel_arg_type_qual !5

**NOTE**: All metadata from the example above are optional. Access qualifiers
are translated for image types, but they should be encoded in LLVM IR type name
rather than function metadata.

Function parameter and global variable decoration through metadata
------------------------------------------------------------------

Both function parameters and global variables can be decorated using LLVM
metadata through the metadata names ``spirv.ParameterDecorations`` and
``spirv.Decorations`` respectively. ``spirv.ParameterDecorations`` must be tied
to the kernel function while ``spirv.Decorations`` is tied directly to the
global variable.

A "decoration-node" is a metadata node consisting of one or more operands. The
first operand is an integer literal representing the SPIR-V decoration
identifier. The other operands are either an integer or string literal
representing the remaining extra operands of the corresponding SPIR-V
decoration.

A "decoration-list" is a metadata node consisting of references to zero or more
decoration-nodes.

``spirv.Decorations`` must refer to a decoration-list while
``spirv.ParameterDecorations`` must refer to a metadata node that contains N
references to decoration-lists, where N is the number of arguments of the
function the metadata is tied to.

``spirv.Decorations`` example:

.. code-block:: llvm

  @v = global i32 0, !spirv.Decorations !1
  ...
  !1 = !{!2, !3}               ; decoration-list with two decoration nodes
  !2 = !{i32 22}               ; decoration-node with no extra operands
  !3 = !{i32 41, !"v", i32 0}  ; decoration-node with 2 extra operands

decorates a global variable ``v`` with ``Constant`` and ``LinkageAttributes``
with extra operands ``"v"`` and ``Export`` in SPIR-V.

``spirv.ParameterDecorations`` example:

.. code-block:: llvm

  define spir_kernel void @k(float %a, float %b) #0 !spirv.ParameterDecorations !1
  ...
  !1 = !{!2, !3} ; metadata node with 2 decoration-lists
  !2 = !{}       ; empty decoration-list
  !3 = !{!4}     ; decoration-list with one decoration node
  !4 = !{i32 19} ; decoration-node with no extra operands

decorates the argument ``b`` of ``k`` with ``Restrict`` in SPIR-V while not
adding any decoration to argument ``a``.

Debug information extension
===========================

**TBD**
