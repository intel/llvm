The SPIRV-LLVM-Translator will "lower" some LLVM intrinsic calls to another function or implementation
using one of the following four methods:

Method 1:

   Variation A:

   In transIntrinsicInst in SPIRVWriter, calls to LLVM intrinsics are replaced with a SPIRV ExtInst.
   For example:

        %0 = tail call i32 @llvm.ctlz.i32(i32 %x, i1 true)

   is translated into SPIRV with an OpenCL ExtInst clz:

        6 ExtInst 2 7 1 clz 5

   The code in transIntrinsicInst to do this translation is:
   
          case Intrinsic::ctlz:
          case Intrinsic::cttz: {
            SPIRVWord ExtOp = IID == Intrinsic::ctlz ? OpenCLLIB::Clz : OpenCLLIB::Ctz;
            SPIRVType *Ty = transType(II->getType());
            std::vector<SPIRVValue *> Ops(1, transValue(II->getArgOperand(0), BB));
            return BM->addExtInst(Ty, BM->getExtInstSetId(SPIRVEIS_OpenCL), ExtOp, Ops,
                                  BB);
          }

   When these ExtInst are reverse translated with (llvm-spirv -r) they are converted to calls:

        %0 = call spir_func i32 @_Z3clzi(i32 %x) #0

   Implementation of the spir_func is in an OpenCL library.  

   If reverse translation is done with (llvm-spirv -r --spirv-target-env=SPV-IR) the calls are converted to
   SPIRV Friendly IR:

        %0 = call spir_func i32 @_Z15__spirv_ocl_clzi(i32 %x)

   This is the cleanest method of lowering.  If a LLVM intrinsic naturally maps to a SPIRV instruction, and if there is an
   external library that supports the instructions this way should be chosen.

   -----------------------------------------------------------------------------------------------------------------------------------
   Varation B:

   Sometimes an intrinsic can be translated to an instruction that is only available with an extension.  For example translating:

        %ret = call i8 @llvm.bitreverse.i8(i8 %a)

   when llvm-spirv is invoked with:

        llvm-spirv --spirv-ext=+SPV_KHR_bit_instructions

   is translated into SPIRV:

        4 BitReverse 51 66 58

   The code in transIntrinsicInst to do this translation is:

          case Intrinsic::bitreverse: {
            if (!BM->getErrorLog().checkError(
                    BM->isAllowedToUseExtension(ExtensionID::SPV_KHR_bit_instructions),
                    SPIRVEC_InvalidFunctionCall, II,
                    "Translation of llvm.bitreverse intrinsic requires "
                    "SPV_KHR_bit_instructions extension.")) {
              return nullptr;
            }
            SPIRVType *Ty = transType(II->getType());
            SPIRVValue *Op = transValue(II->getArgOperand(0), BB);
            return BM->addUnaryInst(OpBitReverse, Ty, Op, BB);
          }

Method 2:

   Some intrinsics are emulated by basic operations in SPIRVWriter.  For example:

        %0 = call float @llvm.vector.reduce.fadd.v4float(float %sp, <4 x float> %v)

   is emulated in SPIRV with:

        5 VectorExtractDynamic 2 11 7 10
        5 VectorExtractDynamic 2 13 7 12
        5 VectorExtractDynamic 2 15 7 14
        5 VectorExtractDynamic 2 17 7 16
        5 FAdd 2 18 6 11
        5 FAdd 2 19 18 13
        5 FAdd 2 20 19 15
        5 FAdd 2 21 20

   The code in transIntrinsicInst to do this emulation is:

        case Intrinsic::vector_reduce_add: {
          Op Op;
          if (IID == Intrinsic::vector_reduce_add) {
            Op = OpIAdd;
          }
          VectorType *VecTy = cast<VectorType>(II->getArgOperand(0)->getType());
          SPIRVValue *VecSVal = transValue(II->getArgOperand(0), BB);
          SPIRVTypeInt *ResultSType =
              BM->addIntegerType(VecTy->getElementType()->getIntegerBitWidth());
          SPIRVTypeInt *I32STy = BM->addIntegerType(32);
          unsigned VecSize = VecTy->getElementCount().getFixedValue();
          SmallVector<SPIRVValue *, 16> Extracts(VecSize);
          for (unsigned Idx = 0; Idx < VecSize; ++Idx) {
            Extracts[Idx] = BM->addVectorExtractDynamicInst(
                VecSVal, BM->addIntegerConstant(I32STy, Idx), BB);
          }
          unsigned Counter = VecSize >> 1;
          while (Counter != 0) {
            for (unsigned Idx = 0; Idx < Counter; ++Idx) {
              Extracts[Idx] = BM->addBinaryInst(Op, ResultSType, Extracts[Idx << 1],
                                                Extracts[(Idx << 1) + 1], BB);
            }
            Counter >>= 1;
          }
          if ((VecSize & 1) != 0) {
            Extracts[0] = BM->addBinaryInst(Op, ResultSType, Extracts[0],
                                            Extracts[VecSize - 1], BB);
          }
          return Extracts[0];
        }


Method 3:

   In SPIRVRegularizeLLVMPass, calls to LLVM intrinsics are replaced with a call to an emulation function.
   The emulation function is created by LLVM API calls and will be translated to SPIRV. The calls to the emulation
   functions and the emulation functions themselves will be translated to SPIRV.  After reverse translation, the calls to the emulation
   functions and the emulation functions themselves will appear in the LLVM IR.

   For example, calls to llvm.bswap.i16:

        %ret = call i16 @llvm.bswap.i16(i16 %0)

   will be re-directed to an emulation function:

        %ret = call i16 @spirv.llvm_bswap_i16(i16 %0)

   The emulation function is constructed by the translator in SPIRVRegularizeLLVM (note that this code
   handles all types):

            case Intrinsic::bswap: {
              BasicBlock *EntryBB = BasicBlock::Create(M->getContext(), "entry", F);
              IRBuilder<> IRB(EntryBB);
              auto *BSwap = IRB.CreateIntrinsic(Intrinsic::bswap, Intrinsic->getType(),
                                                F->getArg(0));
              IRB.CreateRet(BSwap);
              IntrinsicLowering IL(M->getDataLayout());
              IL.LowerIntrinsicCall(BSwap);
              break;
            }

   This will produce a function like:

        define i16 @spirv.llvm_bswap_i16(i16 %0) {
        entry:
          %bswap.2 = shl i16 %0, 8
          %bswap.1 = lshr i16 %0, 8
          %bswap.i16 = or i16 %bswap.2, %bswap.1
          ret i16 %bswap.i16
        }

   After forward translation the emulation calls and functions will appear in SPIRV:

        8 Name 24 "spirv.llvm_bswap_i16"
        ...
        5 FunctionCall 8 26 24 22
        ...
        5 Function 8 24 0 23
        3 FunctionParameter 8 25
        2 Label 42
        5 ShiftLeftLogical 8 44 25 43
        5 ShiftRightLogical 8 45 25 43
        5 BitwiseOr 8 46 44 45
        2 ReturnValue 46

   In reverse translation, the lowering is undone.  Calls are reverted to the original llvm.bswap.i16 intrinsic

        %ret = call i16 @llvm.bswap.i16(i16 %0)

   The emulation functions are deleted.

   The functionality of the intrinsic is created by a call to LLVM's CreateIntrinsic, so the complexity within the
   translator is small.  However, this is effectively using the translator to insert a library function at translation
   time.

Method 4:

   In SPIRVLowerLLVMIntrinsicPass, calls to LLVM intrinsics are replaced with a call to an emulation function.
   The emulation function is represented as a text string of LLVM assembly and is parsed and added to the LLVM IR
   to be translated.  The calls to the emulation functions and the emulation functions themselves will be translated
   to SPIRV.  After reverse translation, the calls to the emulation functions and the emulation functions themselves will appear
   in the LLVM IR.

   For example if SPV_KHR_bit_instructions is not enabled then bit instructions are not supported and llvm.bitreverse.i8
   will be emulated (Note that this is the same intrinsic example used in section 1.B). Calls to it:

        %ret = call i8 @llvm.bitreverse.i8(i8 %a)

   will be re-directed to an emulation function:

        %ret = call i8 @llvm_bitreverse_i8(i8 %a)

   The emulation function is built into the translator.  The source is recorded as a string in LLVMBitreverse.h (note that a separate
   emulation function is needed for each type):

        static const char LLVMBitreversei8[]{R"(
        define zeroext i8 @llvm_bitreverse_i8(i8 %A) {
        entry:
          %and = shl i8 %A, 4
          %shr = lshr i8 %A, 4
          %or = or disjoint i8 %and, %shr
          %and5 = shl i8 %or, 2
          %shl6 = and i8 %and5, -52
          %shr8 = lshr i8 %or, 2
          %and9 = and i8 %shr8, 51
          %or10 = or disjoint i8 %shl6, %and9
          %and13 = shl i8 %or10, 1
          %shl14 = and i8 %and13, -86
          %shr16 = lshr i8 %or10, 1
          %and17 = and i8 %shr16, 85
          %or18 = or disjoint i8 %shl14, %and17
          ret i8 %or18
        }
        )"};

   The supported lowerings are recorded in a table in SPIRVLowerLLVMIntrinsic:

          //  LLVM Intrinsic Name             Required Extension                                   Forbidden Extension                    Module with
          //                                                                                                                              emulation function
            ...
            { "llvm.bitreverse.i8",          {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversei8}},
            ...


   This is the most flexible way of lowering, but it requires a lot of work to create emulation functions for all the necesary types.
   The functionality of a call is provided by LLVM IR supplied as text.  The complexity of the intrinsic functionality is inside the translator.
   Each function signature variation for an intrinsic that needs to be lowered must be supplied by the developer.  As with #2
   this method is effectively using the translator to insert a library function at translation time.
