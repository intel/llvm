SYCL INTEL spatial pipes
========================

Introduction
============

SPIR-V is first class target in which SYCL pipes should be representable, and
pipes are already exposed within SPIR-V. For this implementation API functions
call for SPIR-V friendly mangled functions instead of OpenCL built-ins.
This document describes how SYCL pipes are being lowered to SPIR-V.

OpenCL 2.2 program pipe representation in SPIR-V
================================================

The SPIR-V program pipe representation is used to be an underlying
representation of intra-kernel and inter-kernel static pipe connectivity.
The SPIR-V pipe representation exists in a series of pieces:

 - OpTypePipeStorage: Type representing memory allocated for storage of data
 within a pipe. Used for OpenCL 2.2 program pipes (program-scope pipes) that
 the host program is not aware of, but that enables connectivity between
 kernels.

 - OpConstantPipeStorage: Instruction that creates an OpTypePipeStorage object.
 Requires packet size (number of bytes) and capacity (number of packets) to be
 defined.

 - OpTypePipe: A pipe object that can act as a read/write endpoint of some pipe
 storage, either allocated by the host and passed as a kernel argument, or
 allocated at "program scope" through a pipe storage object.

 - OpCreatePipeFromPipeStorage: Creates a pipe object (that can be read/written)
 from an OpTypePipeStorage instance.

 - OpReadPipe / OpWritePipe: Read packet from or write packet to a pipe object.

Lowering of kernel to kernel pipes to SPIR-V (non-blocking)
===========================================================

This connectivity is achieved through OpTypePipeStorage which allows a SPIR-V
device consumer to leverage static connectivity. An OpConstantPipeStorage
instruction must create a single instance of OpPipeStorage for each kernel to
kernel pipe type used by any kernel within the application.

OpTypePipe objects is created from OpPipeStorage using
OpCreatePipeFromPipeStorage. The number of OpTypePipe objects created from an
OpPipeStorage object is an implementation detail, as are the access qualifiers
applied to those types. For example, an implementation is free to create a
different OpTypePipe corresponding to each read and write, with unidirectional
access qualifiers annotated, or it can create fewer OpTypePipe objects, although
read and write pipes must be distinct according to OpReadPipe and OpWritePipe
rules.

NOTE: The SPIR-V OpReadPipe and OpWritePipe instructions are non-blocking.

Details SPIR-V representation in LLVM IR
========================================

Pipe built-ins are mangled in LLVM IR to make it SPIR-V friendly.
As an example:

 SPIR-V built-in             | Mangled built-in in LLVM IR
 ----------------------------+-----------------------------------------------
 OpReadPipe                  | __spirv_ReadPipe
 ----------------------------+-----------------------------------------------
 OpWritePipe                 | __spirv_WritePipe
 ----------------------------+-----------------------------------------------
 OpCreatePipeFromPipeStorage | __spirv_CreatePipeFromPipeStorage_{read|write}

More about SPIR-V representation in LLVM IR can be found under the link:
.. _SPIRVRepresentationInLLVM.rst: https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst/

In SYCL headers the built-ins are declared as external functions with the
appropriate mangling. The translator will transform calls of these built-ins
into calls of SPIR-V instructions.

Example of SYCL -> LLVM IR -> SPIR-V -> LLVM-IR code transformations
====================================================================
Consider following SYCL device code:
.. code:: cpp
  pipe<class some_pipe, int, 1>::write(42, SuccessCode);

After compiling this code with clang we will be given following piece of IR for
the write pipe function call (NOTE: for this implementation clang-known
OpenCL 2.0 pipe types are reused):
.. code:: cpp
  define internal spir_func void @_ZN2cl4sycl4pipeIZ4mainE9some_pipeiLi1EE5writeEiRb(i32, i8* dereferenceable(1)) #4 align 2 {
    //...
    %12 = call spir_func i32 @_Z17__spirv_WritePipeIiEi8ocl_pipePT_ii(%opencl.pipe_wo_t addrspace(1)* %10, i32 addrspace(4)* %11, i32 4, i32 4) #8
    //...
  }

with following declaration:
.. code:: cpp
  %12 = call spir_func i32 @_Z17__spirv_WritePipeIiEi8ocl_pipePT_ii(%opencl.pipe_wo_t addrspace(1)* %10, i32 addrspace(4)* %11, i32 4, i32 4) #8

SPIR-V translator will drop all of these manglings, just making a call of SPIR-V
write pipe built-in:
.. code:: cpp
 7 WritePipe 51 158 156 157 52 52

Resulting code for translation back to LLVM IR from SPIR-V are calls of OpenCL
built-ins:
.. code:: cpp
  define internal spir_func void @_ZN2cl4sycl4pipeIZ4mainE9some_pipeiLi1EE5writeEiRb(i32, i8*) #0 {
    //...
    %9 = call spir_func i32 @__write_pipe_2(%opencl.pipe_wo_t addrspace(1)* %6, i8 addrspace(4)* %8, i32 4, i32 4)    //...
  }

again with write pipe declaration (but now it's built-in!):
.. code:: cpp
  declare spir_func i32 @__write_pipe_2(%opencl.pipe_wo_t addrspace(1)*, i8 addrspace(4)*, i32, i32) #0

The first argument in a call of __write_pipe_2 OpenCL built-in is a pipe object,
which is created as a result of SPIR-V built-in call
__spirv_CreatePipeFromPipeStorage_{read|write} which has no OpenCL
representation and therefore stays in IR before and after SPIR-V tool-chain as:
.. code:: cpp
  %9 = call spir_func %opencl.pipe_wo_t addrspace(1)* @_Z39__spirv_CreatePipeFromPipeStorage_writeIiE8ocl_pipe11PipeStorage(%struct._ZTS11PipeStorage.PipeStorage* byval align 4 %6) #8
