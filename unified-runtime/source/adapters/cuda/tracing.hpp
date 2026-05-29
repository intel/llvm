//===--------- tracing.hpp - CUDA Host API Tracing -------------------------==//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
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
