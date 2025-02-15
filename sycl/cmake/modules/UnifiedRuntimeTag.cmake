# commit 692bb0f6cf013935d2d5f6244af5e63507ade007 (tag: llvm-test-tag, origin/llvm-build, llvm-build)
# Author: Lukasz Dorau <lukasz.dorau@intel.com>
# Date:   Tue Feb 4 14:07:58 2025 +0100
#
#    Use UMF Proxy pool manager with UMF CUDA memory provider in UR
#
#    UMF Proxy pool manager is just a wrapper for the UMF memory provider
#    (CUDA memory provider in this case) plus it adds also tracking of memory
#    allocations.
#
#    Signed-off-by: Lukasz Dorau <lukasz.dorau@intel.com>
#
set(UNIFIED_RUNTIME_TAG llvm-test-tag)
