SYCL INTEL spatial pipes
========================

Introduction
============

Pipe is a memory object that stores data organized as a FIFO buffer.
This implementation enables 2 classes of pipe connectivity:
  Cross kernel: Kernel A -> Kernel B
  Intra-kernel: Kernel A -> Kernel A

and 2 types of pipes: non-blocking and blocking.

Blocking calls wait until there is available capacity to commit data, or until
data is available to be read. In all of these cases the pipe indirectly conveys
side channel control information to the program, which can be used for flow
control or many other purposes by an application.

Unsuccessful non-blocking pipe reads or writes do not impact the state or
content of a pipe.

Pipe identification
===================

Identification mechanism of a pipe is base on template type specialization of
pipe class.

Consider following code:
.. code:: cpp
  template <class name, typename dataT, size_t min_capacity = 0> class pipe;

where 'name' is a name of a pipe that helps to identify pipe;
      'dataT' is a type of data to store into the pipe, it is required to have
      a default constructor, a copy constructor and a destructor.
      'min_capacity' is the number of outstanding words that can be written to
                     a pipe but not yet read from it.

The combined set of the three template parameters forms the type of a pipe.
Any uses of a read/write method on that type operate on the same pipe by
static connectivity of pipes.

Interaction with SYCL API
=========================

Following member functions of pipe class are available for user:
.. code:: cpp
  // Non-blocking pipes
  static dataT read(bool &success_code);
  static void write(const dataT &data, bool &success_code);

  // Blocking pipes
  static dataT read();
  static void write(const dataT &data);

Writes to or reads from a pipe are accesses to a pipe with the same pipe type.

Simple example of SYCL program
==============================
Non-blocking pipes:
.. code:: cpp
  using Pipe = pipe<class some_pipe, int, 1>;
  Queue.submit([&](handler& cgh) {
    auto read_acc = readBuf.get_access<access::mode::read>(cgh);
    cgh.single_task<class foo_nb>([=]() {
      bool SuccessCode;
      do {
        Pipe::write(read_add[0], SuccessCode); // Write into a some_pipe
                                                 // allocated for integers
                                                 // with a capacity of 1.
      } while (!SuccessCode);
    });
  });
  buffer<int, 1> writeBuf (data, range<1>(dataSize));
  Queue.submit([&](handler& cgh) {
    auto write_acc = writeBuf.get_access<access::mode::write>(cgh);
    cgh.single_task<class goo_nb>([=]() {
      bool SuccessCode;
      do {
        write_acc[0] = Pipe::read(SuccessCode); // Read data stored in the
                                                  // pipe and put it in the
                                                  // SYCL buffer.
      } while (!SuccessCode);
    });
  });

Blocking pipes:
.. code:: cpp
  using Pipe = pipe<class some_pipe, int, 1>;
  Queue.submit([&](handler& cgh) {
    auto read_acc = readBuf.get_access<access::mode::read>(cgh);
    cgh.single_task<class foo_nb>([=]() {
      Pipe::write(read_add[0]); // Write '42' into a some_pipe allocated
                                  // for integers with a capacity of 1.
    });
  });
  buffer<int, 1> writeBuf (data, range<1>(dataSize));
  Queue.submit([&](handler& cgh) {
    auto write_acc = writeBuf.get_access<access::mode::write>(cgh);
    cgh.single_task<class goo_nb>([=]() {
      write_acc[0] = Pipe::read(); // Read data stored in the
                                     // pipe and put it in the
                                     // SYCL buffer.
    });
  });
