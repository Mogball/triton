//
// After tritongpu-pipeline
//

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg2: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg3: f32, %arg4: i32, %arg5: !tt.ptr<f32>) {
    %cst = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #blocked>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = ub.poison : !ttg.async.token
    %c3_i32 = arith.constant 3 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_2, %token_3 = ttng.tmem_alloc : () -> (!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %1 = ttg.memdesc_subview %result_2[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %2 = ttng.tmem_store %cst_0, %1[%token_3], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %5 = ttg.memdesc_subview %4[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %6 = ttg.memdesc_subview %4[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %6, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %7 = ttg.memdesc_subview %4[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %9 = ttg.memdesc_subview %8[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %10 = ttg.memdesc_subview %8[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %10, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %11 = ttg.memdesc_subview %8[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %11, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %5, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %6, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %7, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>
    %13 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %14 = ttg.memdesc_subview %13[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %14, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %15 = ttg.memdesc_subview %13[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %15, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %16 = ttg.memdesc_subview %13[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %16, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %17 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %18 = ttg.memdesc_subview %17[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %18, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %19 = ttg.memdesc_subview %17[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %19, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %20 = ttg.memdesc_subview %17[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %20, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %14, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %15, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %16, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %22 = ttg.memdesc_subview %21[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %22, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %23 = ttg.memdesc_subview %21[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %23, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %24 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %25 = ttg.memdesc_subview %24[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %25, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %26 = ttg.memdesc_subview %24[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %26, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %25, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %26, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %27 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %28 = ttg.memdesc_subview %27[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %28, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %29 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %30 = ttg.memdesc_subview %29[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %30, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %30, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %28, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result_4 = ttng.tmem_alloc : () -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>
    %31 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %32 = ttg.memdesc_subview %31[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %32, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %33 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %34 = ttg.memdesc_subview %33[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %34, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %31, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %35 = ttg.local_alloc : () -> !ttg.memdesc<3x256xf32, #shared1, #smem, mutable>
    %36 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %37 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %38 = ttg.memdesc_subview %35[%c0_i32, %c0_i32] : !ttg.memdesc<3x256xf32, #shared1, #smem, mutable> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256>
    ttg.local_store %cst_1, %38 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256>
    %39 = ttg.memdesc_subview %36[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %40 = ttg.memdesc_subview %37[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %39, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %40, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %39, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %41 = ttg.memdesc_subview %36[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %42 = ttg.memdesc_subview %37[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %41, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %42, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %42, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %43 = ttg.memdesc_subview %36[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %44 = ttg.memdesc_subview %37[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %43, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %44, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %44, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %45:6 = ttg.warp_specialize(%4, %8, %3, %result, %24, %21, %arg0, %35, %36, %37, %arg5, %13, %17, %12, %28, %33, %result_4, %1, %30, %31, %arg4, %arg1, %arg2) attributes {requestedRegisters = array<i32: 24, 24, 88>}
    default {
      %47:8 = scf.for %arg6 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg7 = %cst, %arg8 = %cst_1, %arg9 = %2, %arg10 = %c0_i32, %arg11 = %c0_i32, %arg12 = %c0_i32, %arg13 = %c0_i32, %arg14 = %c0_i32) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, i32, i32, i32, i32, i32)  : i32 {
        %48 = ttg.memdesc_subview %result[%arg10, %c0_i32, %c0_i32] : !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
        %49 = ttg.memdesc_subview %24[%arg10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %50 = ttg.memdesc_subview %21[%arg10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %50, %arg11 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %result_7, %token_8 = ttng.tmem_load %48[] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64> -> tensor<256x64xf32, #blocked>
        ttng.arrive_barrier %49, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %51 = arith.addi %arg10, %c1_i32 : i32
        %52 = arith.xori %arg11, %c1_i32 : i32
        %53 = arith.cmpi eq, %51, %c2_i32 : i32
        %54 = arith.select %53, %c0_i32, %51 : i32
        %55 = arith.select %53, %52, %arg11 : i32
        %56 = "compute_row_max"(%result_7, %arg3) : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %57 = arith.addi %arg13, %c1_i32 : i32
        %58 = arith.xori %arg14, %c1_i32 : i32
        %59 = arith.cmpi eq, %57, %c3_i32 : i32
        %60 = arith.select %59, %58, %arg14 : i32
        %61 = arith.select %59, %c1_i32, %57 : i32
        %62 = ttg.memdesc_subview %35[%61, %c0_i32] : !ttg.memdesc<3x256xf32, #shared1, #smem, mutable> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256>
        %63 = ttg.memdesc_subview %36[%61] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %64 = ttg.memdesc_subview %37[%61] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %64, %60 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttg.local_store %56, %62 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256>
        ttng.arrive_barrier %63, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %65 = "sub_row_max"(%result_7, %56, %arg3) : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
        %66 = math.exp2 %65 : tensor<256x64xf32, #blocked>
        %67 = arith.subf %arg8, %56 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %68 = math.exp2 %67 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %69 = "tt.reduce"(%66) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %74 = arith.addf %arg15, %arg16 : f32
          tt.reduce.return %74 : f32
        }) : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %70 = arith.mulf %arg7, %68 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %71 = arith.addf %70, %69 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %72 = arith.truncf %66 : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
        ttng.wait_barrier %31, %arg12 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tmem_store %72, %result_4, %true : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %33, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %73 = arith.xori %arg12, %c1_i32 : i32
        scf.yield %71, %56, %0, %54, %55, %73, %61, %60 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, i32, i32, i32, i32, i32
      } {tt.warp_specialize}
      ttg.warp_yield %47#0, %47#1, %47#2, %47#3, %47#4, %47#5 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, i32, i32, i32
    }
    partition0(%arg6: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg7: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg8: !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>, %arg9: !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>, %arg10: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg11: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg12: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg13: !ttg.memdesc<3x256xf32, #shared1, #smem, mutable>, %arg14: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg15: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg16: !tt.ptr<f32>, %arg17: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg18: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg19: !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>, %arg20: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg21: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg22: !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>, %arg23: !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, %arg24: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg25: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg26: i32, %arg27: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg28: !tt.tensordesc<tensor<64x64xf16, #shared>>) num_warps(1) {
      %c128_i32 = arith.constant 128 : i32
      %c64_i32_7 = arith.constant 64 : i32
      %c0_i32_8 = arith.constant 0 : i32
      %false = arith.constant false
      %true_9 = arith.constant true
      %c1_i32_10 = arith.constant 1 : i32
      %c2_i32_11 = arith.constant 2 : i32
      %c3_i32_12 = arith.constant 3 : i32

      // Peeled MMA QK0
      %47 = arith.cmpi sgt, %arg26, %c0_i32_8 : i32
      %48 = ttg.memdesc_subview %arg6[%c0_i32_8] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %49 = ttg.memdesc_subview %arg7[%c0_i32_8] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %50 = ttg.memdesc_subview %arg8[%c0_i32_8, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
      %51 = ttg.memdesc_trans %50 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable, 3x64x64>
      ttng.wait_barrier %49, %c0_i32_8, %47 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %52 = ttg.memdesc_subview %arg9[%c0_i32_8, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %53 = ttg.memdesc_subview %arg10[%c0_i32_8] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %53, %c0_i32_8, %47 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %54 = ttg.memdesc_subview %arg11[%c0_i32_8] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %55 = ttng.tc_gen5_mma %arg12, %51, %52[], %false, %47, %48[%true_9], %54[%true_9] {tt.self_latency = 1 : i32} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared2, #smem, mutable, 3x64x64>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>

      // Peeled MMA QK1
      %56 = arith.cmpi sgt, %arg26, %c64_i32_7 : i32
      %57 = ttg.memdesc_subview %arg6[%c1_i32_10] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %58 = ttg.memdesc_subview %arg7[%c1_i32_10] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %59 = ttg.memdesc_subview %arg8[%c1_i32_10, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
      %60 = ttg.memdesc_trans %59 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable, 3x64x64>
      ttng.wait_barrier %58, %c0_i32_8, %56 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %61 = ttg.memdesc_subview %arg9[%c1_i32_10, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %62 = ttg.memdesc_subview %arg10[%c1_i32_10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %62, %c0_i32_8, %56 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %63 = ttg.memdesc_subview %arg11[%c1_i32_10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %64 = ttng.tc_gen5_mma %arg12, %60, %61[], %false, %56, %57[%true_9], %63[%true_9] {tt.self_latency = 1 : i32} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared2, #smem, mutable, 3x64x64>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>

      %65:7 = scf.for %arg29 = %c0_i32_8 to %arg26 step %c64_i32_7 iter_args(%arg30 = %c1_i32_10, %arg31 = %c0_i32_8, %arg32 = %c1_i32_10, %arg33 = %c0_i32_8, %arg34 = %c0_i32_8, %arg35 = %c0_i32_8, %arg36 = %c0_i32_8) -> (i32, i32, i32, i32, i32, i32, i32)  : i32 {
        %66 = arith.subi %arg26, %c128_i32 : i32
        %67 = arith.cmpi slt, %arg29, %66 : i32
        %68 = arith.xori %arg34, %c1_i32_10 : i32

        // MMA PV(i)
        %69 = ttg.memdesc_subview %arg17[%arg35] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %70 = ttg.memdesc_subview %arg18[%arg35] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %71 = ttg.memdesc_subview %arg19[%arg35, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
        ttng.wait_barrier %70, %arg36 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %arg20, %68, %true_9 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %arg21, %arg34 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %72 = ttng.tc_gen5_mma %arg22, %71, %arg23[], %true_9, %true_9, %69[%true_9], %arg24[%true_9], %arg25[%true_9] {tt.self_latency = 1 : i32} : !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %73 = arith.addi %arg32, %c1_i32_10 : i32
        %74 = arith.xori %arg33, %c1_i32_10 : i32
        %75 = arith.cmpi eq, %73, %c2_i32_11 : i32
        %76 = arith.select %75, %c0_i32_8, %73 : i32
        %77 = arith.select %75, %74, %arg33 : i32
        %78 = arith.addi %arg30, %c1_i32_10 : i32
        %79 = arith.xori %arg31, %c1_i32_10 : i32
        %80 = arith.cmpi eq, %78, %c3_i32_12 : i32
        %81 = arith.select %80, %c0_i32_8, %78 : i32
        %82 = arith.select %80, %79, %arg31 : i32

        // MMA PV(i+2) masked for last 2 iterations
        %83 = ttg.memdesc_subview %arg6[%81] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %84 = ttg.memdesc_subview %arg7[%81] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %85 = ttg.memdesc_subview %arg8[%81, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
        %86 = ttg.memdesc_trans %85 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable, 3x64x64>
        ttng.wait_barrier %84, %82, %67 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %87 = ttg.memdesc_subview %arg9[%76, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
        %88 = ttg.memdesc_subview %arg10[%76] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %88, %77, %67 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %89 = ttg.memdesc_subview %arg11[%76] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %90 = ttng.tc_gen5_mma %arg12, %86, %87[], %false, %67, %83[%true_9], %89[%true_9] {tt.self_latency = 1 : i32} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared2, #smem, mutable, 3x64x64>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        scf.yield %81, %82, %76, %77, %68, %arg30, %arg31 : i32, i32, i32, i32, i32, i32, i32
      } {tt.warp_specialize}
      ttg.warp_return
    }
    partition1(%arg6: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg7: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg8: !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>, %arg9: !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>, %arg10: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg11: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg12: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg13: !ttg.memdesc<3x256xf32, #shared1, #smem, mutable>, %arg14: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg15: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg16: !tt.ptr<f32>, %arg17: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg18: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg19: !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>, %arg20: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg21: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg22: !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>, %arg23: !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, %arg24: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg25: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg26: i32, %arg27: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg28: !tt.tensordesc<tensor<64x64xf16, #shared>>) num_warps(2) {
      %c128_i32 = arith.constant 128 : i32
      %c64_i32_7 = arith.constant 64 : i32
      %c0_i32_8 = arith.constant 0 : i32
      %true_9 = arith.constant true
      %c1_i32_10 = arith.constant 1 : i32
      %c3_i32_11 = arith.constant 3 : i32

      // Peeled load K0
      %47 = arith.cmpi sgt, %arg26, %c0_i32_8 : i32
      %48 = ttg.memdesc_subview %arg6[%c0_i32_8] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %48, %c0_i32_8, %47 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %49 = ttg.memdesc_subview %arg7[%c0_i32_8] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.barrier_expect %49, 8192, %47 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %50 = ttg.memdesc_subview %arg8[%c0_i32_8, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
      ttng.async_tma_copy_global_to_local %arg27[%c0_i32_8, %c0_i32_8] %50, %49, %47 : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>

      // Peeled load K1
      %51 = arith.cmpi sgt, %arg26, %c64_i32_7 : i32
      %52 = ttg.memdesc_subview %arg6[%c1_i32_10] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %52, %c0_i32_8, %51 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %53 = ttg.memdesc_subview %arg7[%c1_i32_10] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.barrier_expect %53, 8192, %51 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %54 = ttg.memdesc_subview %arg8[%c1_i32_10, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
      ttng.async_tma_copy_global_to_local %arg27[%c64_i32_7, %c0_i32_8] %54, %53, %51 : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>

      %55:4 = scf.for %arg29 = %c0_i32_8 to %arg26 step %c64_i32_7 iter_args(%arg30 = %c1_i32_10, %arg31 = %c0_i32_8, %arg32 = %c0_i32_8, %arg33 = %c0_i32_8) -> (i32, i32, i32, i32)  : i32 {
        %56 = arith.subi %arg26, %c128_i32 : i32
        %57 = arith.cmpi slt, %arg29, %56 : i32

        // Load V(i)
        %58 = ttg.memdesc_subview %arg17[%arg32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %58, %arg33 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %59 = ttg.memdesc_subview %arg18[%arg32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.barrier_expect %59, 8192, %true_9 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %60 = ttg.memdesc_subview %arg19[%arg32, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
        ttng.async_tma_copy_global_to_local %arg28[%arg29, %c0_i32_8] %60, %59, %true_9 : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
        %61 = arith.addi %arg30, %c1_i32_10 : i32
        %62 = arith.xori %arg31, %c1_i32_10 : i32
        %63 = arith.cmpi eq, %61, %c3_i32_11 : i32
        %64 = arith.select %63, %c0_i32_8, %61 : i32
        %65 = arith.select %63, %62, %arg31 : i32

        // Load V(i+2) masked for last 2 iterations
        %66 = ttg.memdesc_subview %arg6[%64] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %66, %65, %57 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %67 = ttg.memdesc_subview %arg7[%64] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.barrier_expect %67, 8192, %57 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %68 = ttg.memdesc_subview %arg8[%64, %c0_i32_8, %c0_i32_8] : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
        %69 = arith.addi %arg29, %c128_i32 : i32
        ttng.async_tma_copy_global_to_local %arg27[%69, %c0_i32_8] %68, %67, %57 : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
        scf.yield %64, %65, %arg30, %arg31 : i32, i32, i32, i32
      } {tt.warp_specialize}
      ttg.warp_return
    }
    partition2(%arg6: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg7: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg8: !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>, %arg9: !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>, %arg10: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg11: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg12: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg13: !ttg.memdesc<3x256xf32, #shared1, #smem, mutable>, %arg14: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg15: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg16: !tt.ptr<f32>, %arg17: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg18: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg19: !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>, %arg20: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg21: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg22: !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>, %arg23: !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, %arg24: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg25: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg26: i32, %arg27: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg28: !tt.tensordesc<tensor<64x64xf16, #shared>>) num_warps(4) {
      %c64_i32_7 = arith.constant 64 : i32
      %c0_i32_8 = arith.constant 0 : i32
      %true_9 = arith.constant true
      %c1_i32_10 = arith.constant 1 : i32
      %c3_i32_11 = arith.constant 3 : i32
      %c-1_i32 = arith.constant -1 : i32
      %47:5 = scf.for %arg29 = %c0_i32_8 to %arg26 step %c64_i32_7 iter_args(%arg30 = %c0_i32_8, %arg31 = %c0_i32_8, %arg32 = %c0_i32_8, %arg33 = %c-1_i32, %arg34 = %c0_i32_8) -> (i32, i32, i32, i32, i32)  : i32 {
        %48 = arith.addi %arg31, %c1_i32_10 : i32
        %49 = arith.xori %arg32, %c1_i32_10 : i32
        %50 = arith.cmpi eq, %48, %c3_i32_11 : i32
        %51 = arith.select %50, %49, %arg32 : i32
        %52 = arith.select %50, %c1_i32_10, %48 : i32
        %53 = ttg.memdesc_subview %arg13[%52, %c0_i32_8] : !ttg.memdesc<3x256xf32, #shared1, #smem, mutable> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256>
        %54 = ttg.memdesc_subview %arg14[%52] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %55 = ttg.memdesc_subview %arg15[%52] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %56 = tt.addptr %arg16, %arg29 : !tt.ptr<f32>, i32
        %57 = tt.load %56 : !tt.ptr<f32>
        %58 = tt.splat %57 : f32 -> tensor<256x64xf32, #blocked>
        ttng.wait_barrier %54, %51 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %59 = ttg.local_load %53 : !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256> -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        ttng.arrive_barrier %55, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %60 = arith.addi %arg33, %c1_i32_10 : i32
        %61 = arith.xori %arg34, %c1_i32_10 : i32
        %62 = arith.cmpi eq, %60, %c3_i32_11 : i32
        %63 = arith.select %62, %61, %arg34 : i32
        %64 = arith.select %62, %c1_i32_10, %60 : i32
        %65 = ttg.memdesc_subview %arg13[%64, %c0_i32_8] : !ttg.memdesc<3x256xf32, #shared1, #smem, mutable> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256>
        %66 = ttg.memdesc_subview %arg14[%64] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %67 = ttg.memdesc_subview %arg15[%64] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %66, %63 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %68 = ttg.local_load %65 : !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256> -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        ttng.arrive_barrier %67, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %69 = arith.subf %68, %59 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %70 = math.exp2 %69 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %71 = tt.expand_dims %70 {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
        %72 = tt.broadcast %71 : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
        ttng.wait_barrier %arg24, %arg30 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %result_12, %token_13 = ttng.tmem_load %arg23[] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
        %73 = arith.xori %arg30, %c1_i32_10 : i32
        %74 = arith.mulf %result_12, %72 : tensor<256x64xf32, #blocked>
        %75 = arith.addf %74, %58 : tensor<256x64xf32, #blocked>
        %76 = ttng.tmem_store %75, %arg23[], %true_9 : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
        ttng.arrive_barrier %arg20, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        scf.yield %73, %52, %51, %64, %63 : i32, i32, i32, i32, i32
      } {tt.warp_specialize}
      ttg.warp_return
    } : (!ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<3x256xf32, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !tt.ptr<f32>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, i32, !tt.tensordesc<tensor<64x64xf16, #shared>>, !tt.tensordesc<tensor<64x64xf16, #shared>>) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, i32, i32, i32)
    ttg.local_dealloc %35 : !ttg.memdesc<3x256xf32, #shared1, #smem, mutable>
    ttng.inval_barrier %39 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %40 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %41 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %42 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %43 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %44 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %36 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %37 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %30, %45#5 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %34 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %33 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %31 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %30 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %29 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %28 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %27 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %46 = ttg.memdesc_subview %24[%45#3] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.wait_barrier %46, %45#4 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %25 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %26 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %24 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %22 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %23 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %21 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %18 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %19 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %20 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %17 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %14 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %15 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %16 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %13 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %12 : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>
    ttng.inval_barrier %9 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %10 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %11 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %8 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %5 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %6 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %7 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %4 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %3 : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>
    %result_5, %token_6 = ttng.tmem_load %1[%45#2] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
    "use"(%45#0, %result_5, %45#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}

