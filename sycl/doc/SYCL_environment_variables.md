# Overview

This file describes environment variables that are having effect on SYCL compiler and run-time.

# Controlling SYCL RT

| Environment variable | Description |
| ----------- | ----------- |
| SYCL_PI_TRACE | If set forces tracing of PI calls to stdout. |
| SYCL_BE={PI_OPENCL,PI_OTHER} | When SYCL RT is buils with PI this controls which plugin to use. |
