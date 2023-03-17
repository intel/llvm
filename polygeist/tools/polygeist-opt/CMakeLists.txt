get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRPolygeist
        MLIRPolygeistTransforms
        MLIRSYCLDialect
        MLIRSYCLToGPU
        MLIRSYCLToLLVM
        MLIRSYCLToSPIRV
        MLIRSYCLTransforms
)
add_llvm_executable(polygeist-opt polygeist-opt.cpp)

install(TARGETS polygeist-opt
EXPORT PolygeistTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT polygeist-opt)

llvm_update_compile_flags(polygeist-opt)
target_link_libraries(polygeist-opt PRIVATE ${LIBS})
