IO pipes design
===============

Requirements
------------
 - Device shall be able to distinguish kernel-to-kernel pipes and I/O pipes;
 - No changes shall be in SYCL pipe specification;
 - I/O pipe namings/IDs are provided by a vendor in a separated header, like:
.. code:: cpp
  namespace intelfpga {
    template <unsigned ID>
    struct ethernet_pipe_id {
      static constexpr unsigned id = ID;
    };
    using ethernet_read_pipe =
        cl::sycl::intel::kernel_readable_io_pipe<ethernet_pipe_id<0>, int, 0>;
    using ethernet_write_pipe =
        cl::sycl::intel::kernel_writeable_io_pipe<ethernet_pipe_id<1>, int, 0>;
  }

  Thus, the user interacts only with vendor-defined pipe objects.

Links
-----
.. _Spec: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/SYCL_EXT_INTEL_DATAFLOW_PIPES.asciidoc
.. _Interesting comment from Ronan: https://github.com/intel/llvm/pull/635#discussion_r325851766

Summary
-------
This document describes a design of I/O pipes implementation in SYCL compiler.
It includes changes in SYCL library, clang and SPIR-V to LLVM IR translator.
It adds extra attribute '__attribute__((io_pipe_id(ID)))' that generates
metadata, attached to a pipe storage declaration. By this metadata a backend can
recognize I/O pipe and distinguish different I/O pipes from each other.

There is another notable solution which was proposed by Ronan (see the link
above): don't make any compiler/library changes - just make backend recognizing
I/O pipe by demangling it's name. This proposal wasn't picked, because it will
make backend support of the feature more difficult. So far we already have
two devices' backends that support SYCL_INTEL_data_flow_pipes extension
(Intel FPGA HW and Intel FPGA emulator) and in the future this number may
increase. So efforts put in compiler implementation shall be payed off even
more.

clang
-----
Need to implement additional attribute, that possible to attach to pipe storage
declaration. The attribute shall accept a compile-time known integer argument
(the pipe ID). With the attribute applied, clang generates a metadata attached
the to pipe storage declaration, that contains the I/O pipe ID (argument).

llvm-spirv translator
---------------------
Need to implement additional decoration, that saves the I/O pipe ID information,
that can be collected from a metadata attached to the pipe storage object.

SYCL implementation in headers
------------------------------
Following the spec, we need to add two more classes for pipes:
 - 'kernel_readable_io_pipe'
 - 'kernel_writeable_io_pipe'

 with the same member functions and fields as it is already done for pipe class.

The attribute should be attached to a pipe storage declaration in the headers
and it would be looking like:
.. code:: cpp
  static constexpr int32_t ID = name::id;
  static constexpr struct ConstantPipeStorage
  m_Storage __attribute__((io_pipe_id(ID))) = {m_Size, m_Alignment, m_Capacity};


 where 'name' is some class used in pipe type construction, like:
.. code:: cpp
 using pipe_type = pipe<name, dataT, min_capacity>;

When specific io_pipe_def.h is included in a user's code and 'name' is mapped to
the pipe name defined in this header (for example 'ethernet_pipe_id' structure
defined above) 'name::id' returns the actual I/O pipe ID (compile-time known
integer constant) that is passed as the attribute's argument and used to
identify the I/O pipe in some backend.
