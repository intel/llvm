//===--------- tracing.hpp - CUDA Host API Tracing -------------------------==//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

struct cuda_tracing_context_t_;

cuda_tracing_context_t_ *createCUDATracingContext();
void freeCUDATracingContext(cuda_tracing_context_t_ *Ctx);

bool loadCUDATracingLibrary(cuda_tracing_context_t_ *Ctx);
void unloadCUDATracingLibrary(cuda_tracing_context_t_ *Ctx);

void enableCUDATracing(cuda_tracing_context_t_ *Ctx);
void disableCUDATracing(cuda_tracing_context_t_ *Ctx);

// Deprecated. Will be removed once pi_cuda has been updated to use the variant
// that takes a context pointer.
void enableCUDATracing();
