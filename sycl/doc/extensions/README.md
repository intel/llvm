# SYCL Extensions in DPC++

The DPC++ compiler supports the core [SYCL 2020 specification][1], and the
following Khronos-defined KHR extensions:

* [sycl\_khr\_default\_context][khr1]
* [sycl\_khr\_queue\_empty\_query][khr2]

In addition, DPC++ supports a variety of vendor-defined extensions.
The specifications for these extensions are organized into the following
subdirectories according to their state.

| Directory        | Description                                                   |
|------------------|---------------------------------------------------------------|
|[supported][2]    | Extensions which are fully supported                          |
|[experimental][3] | Extensions which are implemented but may change in the future |
|[deprecated][4]   | Extensions which are supported but will be removed soon       |
|[proposed][5]     | Extensions which proposed but not yet implemented             |
|[removed][6]      | Extensions which used to be supported but are now removed     |

[1]: <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html>
[2]: <supported>
[3]: <experimental>
[4]: <deprecated>
[5]: <proposed>
[6]: <removed>

[khr1]: <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:khr-default-context>
[khr2]: <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:khr-queue-empty-query>
