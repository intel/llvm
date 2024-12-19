# Use LLVM for each `tablegen(LLVM)`
# LLVM_HEADERS_TABLEGEN (which is the default for this directory)
# points to `llvm-min-tblgen`, but we need `llvm-tblgen`
set(LLVM_TABLEGEN_PROJECT LLVM)

set(LLVM_TARGET_DEFINITIONS DeviceConfigFile.td)
tablegen(LLVM DeviceConfigFile.inc -gen-dynamic-tables)
add_public_tablegen_target(DeviceConfigFile)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/DeviceConfigFile.hpp"
              "${CMAKE_CURRENT_BINARY_DIR}/DeviceConfigFile.inc"
        DESTINATION llvm/SYCLLowerIR
        COMPONENT DeviceConfigFile)
