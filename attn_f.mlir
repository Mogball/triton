//
// After tritongpu-load-mma-specialization
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

    // 0 = default partition (softmax)
    // 1 = MMA partition
    // 2 = load partition
    // 3 = correction partition (MMA PV backedge critical path)

    %1:4 = scf.for %arg6 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg7 = %cst_1, %arg8 = %cst, %arg9 = %token, %arg10 = %0) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {

      // [[LOAD PARTITION]]
      // Load K and its local_alloc in Load partition
      %2 = tt.descriptor_load %arg1[%arg6, %c0_i32] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %3 = ttg.local_alloc %2 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>

      // [[MMA PARTITION]]
      // memdesc_trans K and user MMA in MMA partition
      %4 = ttg.memdesc_trans %3 {loop.cluster = 2 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      %5 = ttng.tc_gen5_mma %arg0, %4, %result[%arg9], %false, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

      // [[SOFTMAX PARTITION]]
      // load QK in softmax partition
      %result_6, %token_7 = ttng.tmem_load %result[%5] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
      %6 = "compute_row_max"(%result_6, %arg3) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %7 = "sub_row_max"(%result_6, %6, %arg3) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      // expensive exp2 256x64
      %8 = math.exp2 %7 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256x64xf32, #blocked>
      %10 = arith.subf %arg8, %6 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %12 = math.exp2 %10 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %13 = "tt.reduce"(%8) <{axis = 1 : i32}> ({
      ^bb0(%arg11: f32, %arg12: f32):
        %30 = arith.addf %arg11, %arg12 : f32
        tt.reduce.return %30 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = arith.mulf %arg7, %12 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %15 = arith.addf %14, %13 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = tt.expand_dims %12 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %19 = tt.broadcast %17 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %20 = tt.addptr %arg5, %arg6 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : !tt.ptr<f32>, i32
      %21 = tt.load %20 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : !tt.ptr<f32>
      %22 = tt.splat %21 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : f32 -> tensor<256x64xf32, #blocked>
      // p = p.to(dtype)
      %27 = arith.truncf %8 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      // store P to TMEM for LHS
      %result_10 = ttng.tmem_alloc %27 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>

      // [[CORRECTION PARTITION]]
      // exp2(m_i - m_ij) rematerialized in correction partition
      // to reduce dependencies on the critical path
      %9 = arith.subf %arg8, %6 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %11 = math.exp2 %9 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %16 = tt.expand_dims %11 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %18 = tt.broadcast %16 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      // load O from TMEM
      %result_8, %token_9 = ttng.tmem_load %result_2[%arg10] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
      // o *= alpha
      %23 = arith.mulf %result_8, %18 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked>
      %24 = arith.addf %23, %22 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked>
      // store O to TMEM
      %28 = ttng.tmem_store %24, %result_2[%token_9], %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

      // [[LOAD PARTITION]]
      // Load V and its alloc in Load partition
      %25 = tt.descriptor_load %arg2[%arg6, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %26 = ttg.local_alloc %25 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 2 : i32} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>

      // [[MMA PARTITION]]
      // MMA PV in MMA partition
      %29 = ttng.tc_gen5_mma %result_10, %26, %result_2[%28], %true, %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

      scf.yield %15, %6, %token_7, %29 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 4 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32]}
    %result_4, %token_5 = ttng.tmem_load %result_2[%1#3] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    "use"(%1#0, %result_4, %1#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}

