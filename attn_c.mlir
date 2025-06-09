//
// After tritongpu-assign-latencies
//

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg2: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg3: f32, %arg4: i32, %arg5: !tt.ptr<f32>) {
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #blocked>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_2, %token_3 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst_0, %result_2[%token_3], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %1:4 = scf.for %arg6 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg7 = %cst_1, %arg8 = %cst, %arg9 = %token, %arg10 = %0) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {

      // Load K latency=2
      %2 = tt.descriptor_load %arg1[%arg6, %c0_i32] {tt.latency = 2 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>

      %3 = ttg.local_alloc %2 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>

      // MMA QK latency=2 (double buffered accumulator)
      %5 = ttng.tc_gen5_mma %arg0, %4, %result[%arg9], %false, %true {tt.latency = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_6, %token_7 = ttng.tmem_load %result[%5] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
      %6 = "compute_row_max"(%result_6, %arg3) : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %7 = "sub_row_max"(%result_6, %6, %arg3) : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %8 = math.exp2 %7 : tensor<256x64xf32, #blocked>
      %9 = arith.subf %arg8, %6 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %10 = math.exp2 %9 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %11 = "tt.reduce"(%8) <{axis = 1 : i32}> ({
      ^bb0(%arg11: f32, %arg12: f32):
        %26 = arith.addf %arg11, %arg12 : f32
        tt.reduce.return %26 : f32
      }) : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %12 = arith.mulf %arg7, %10 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %13 = arith.addf %12, %11 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = tt.expand_dims %10 {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %15 = tt.broadcast %14 : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %16 = tt.addptr %arg5, %arg6 : !tt.ptr<f32>, i32
      %17 = tt.load %16 : !tt.ptr<f32>
      %18 = tt.splat %17 : f32 -> tensor<256x64xf32, #blocked>
      %result_8, %token_9 = ttng.tmem_load %result_2[%arg10] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
      %19 = arith.mulf %result_8, %15 : tensor<256x64xf32, #blocked>
      %20 = arith.addf %19, %18 : tensor<256x64xf32, #blocked>

      // Load V latency=2
      %21 = tt.descriptor_load %arg2[%arg6, %c0_i32] {tt.latency = 2 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>

      %22 = ttg.local_alloc %21 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %23 = arith.truncf %8 : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      %result_10 = ttng.tmem_alloc %23 : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>
      %24 = ttng.tmem_store %20, %result_2[%token_9], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

      // MMA PV latency=1 (MMA feeds into itself through correction and can't be multibuffered)
      %25 = ttng.tc_gen5_mma %result_10, %22, %result_2[%24], %true, %true {tt.self_latency = 1 : i32} : !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

      scf.yield %13, %6, %token_7, %25 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize}
    %result_4, %token_5 = ttng.tmem_load %result_2[%1#3] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    "use"(%1#0, %result_4, %1#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}

