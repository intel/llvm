# Overview

This file describes environment variables that are having effect on SYCL compiler and run-time.

# Controlling SYCL RT

| Environment variable | Description |
| ----------- | ----------- |
| SYCL_PI_TRACE | If set forces tracing of PI calls to stdout. |
| SYCL_BE={PI_OPENCL,PI_OTHER} | When SYCL RT is built with PI this controls which plugin to use. |
| SYCL_PRINT_EXECUTION_GRAPH | Print execution graph to DOT text file. Options are described below. |

SYCL_PRINT_EXECUTION_GRAPH can accept one or more comma separated values from table below

| Option | Description |
| ------ | ----------- |
| before_addCG | print graph before addCG method |
| after_addCG | print graph after addCG method |
| before_addCopyBack | print graph before addCopyBack method |
| after_addCopyBack | print graph after addCopyBack method |
| before_addHostAcc | print graph before addHostAccessor method |
| after_addHostAcc | print graph after addHostAccessor method |
| always | print graph before and after each of the above methods |

