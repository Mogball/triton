//
// After tritongpu-load-mma-specialization
// but I reordered the code to make the partitions more readable
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
    %c3_i32 = arith.constant 3 : i32
    %0 = ub.poison : !ttg.async.token
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #blocked>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    // 2x multibuffered QK accumulator
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // 1x multibuffered PV accumulator
    %result_2, %token_3 = ttng.tmem_alloc : () -> (!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %1 = ttg.memdesc_subview %result_2[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    // store dense<0> initial value into PV[0]
    %2 = ttng.tmem_store %cst_0, %1[%token_3], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>

    // 3x multibuffered K
    %3 = ttg.local_alloc : () -> !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>

    // 3x empty barriers for K
    %4 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %5 = ttg.memdesc_subview %4[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %6 = ttg.memdesc_subview %4[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %6, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %7 = ttg.memdesc_subview %4[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>

    // 3x ready barriers for K
    %8 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %9 = ttg.memdesc_subview %8[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %10 = ttg.memdesc_subview %8[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %10, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %11 = ttg.memdesc_subview %8[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %11, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>

    // K is initialized for producer (all buffers empty)
    ttng.arrive_barrier %5, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %6, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %7, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>

    // 3x multibuffered V
    %12 = ttg.local_alloc : () -> !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>

    // 3x empty barriers for V
    %13 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %14 = ttg.memdesc_subview %13[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %14, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %15 = ttg.memdesc_subview %13[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %15, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %16 = ttg.memdesc_subview %13[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %16, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>

    // 3x ready barriers for V
    %17 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %18 = ttg.memdesc_subview %17[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %18, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %19 = ttg.memdesc_subview %17[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %19, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %20 = ttg.memdesc_subview %17[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %20, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>

    // V is initialized for producer (all buffers empty)
    ttng.arrive_barrier %14, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %15, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %16, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>

    // 2x (mma->load) barriers for QK
    %21 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %22 = ttg.memdesc_subview %21[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %22, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %23 = ttg.memdesc_subview %21[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %23, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>

    // 2x (load->mma) barriers for QK
    %24 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %25 = ttg.memdesc_subview %24[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %25, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %26 = ttg.memdesc_subview %24[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %26, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>

    // initialize QK barriers for MMA (which runs first)
    ttng.arrive_barrier %25, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %26, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>

    // 1x ((load,store)->mma) barrier for PV
    %27 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %28 = ttg.memdesc_subview %27[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %28, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    // 1x (mma->(load,store)) barrier for PV
    %29 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %30 = ttg.memdesc_subview %29[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %30, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    // both are initialized: (load,store) waits for its barrier with phase 0 first
    ttng.arrive_barrier %30, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // mma waits for its barrier with phase 1 first, so first move its barrier
    // from phase 0 to phase 1
    ttng.arrive_barrier %28, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    // hoisted TMEM alloc for P
    %result_4 = ttng.tmem_alloc {loop.cluster = 0 : i32, loop.stage = 4 : i32} : () -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>
    // empty barrier for P
    %31 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %32 = ttg.memdesc_subview %31[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %32, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // ready barrier for P
    %33 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %34 = ttg.memdesc_subview %33[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %34, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // mark P as empty
    ttng.arrive_barrier %31, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    %35:13 = scf.for %arg6 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg7 = %cst_1, %arg8 = %cst, %arg9 = %2, %arg10 = %c0_i32, %arg11 = %c0_i32, %arg12 = %c0_i32, %arg13 = %c0_i32, %arg14 = %c0_i32, %arg15 = %c0_i32, %arg16 = %c0_i32, %arg17 = %c0_i32, %arg18 = %c0_i32, %arg19 = %c0_i32) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
      // [[LOAD K]]
      // wait for K buffer to be consumed
      %39 = ttg.memdesc_subview %4[%arg10] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %39, %arg11 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      // issue async load K
      %40 = ttg.memdesc_subview %8[%arg10] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.barrier_expect %40, 8192 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32}, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %41 = ttg.memdesc_subview %3[%arg10, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
      ttng.async_tma_copy_global_to_local %arg1[%arg6, %c0_i32] %41, %40, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>

      // [[MMA QK]]
      // wait for K to be ready
      %42 = ttg.memdesc_trans %41 {loop.cluster = 2 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable, 3x64x64>
      ttng.wait_barrier %40, %arg11 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      // wait for QK buffer to be empty
      %43 = ttg.memdesc_subview %result[%arg14, %c0_i32, %c0_i32] : !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %44 = ttg.memdesc_subview %24[%arg14] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %44, %arg15, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      // issue async MMA QK
      %45 = ttg.memdesc_subview %21[%arg14] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %46 = ttng.tc_gen5_mma %arg0, %42, %43[], %false, %true,
        // signal K empty when complete
        %39[%true],
        // signal QK ready when complete
        %45[%true] {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared2, #smem, mutable, 3x64x64>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>

      // [[SOFTMAX]]
      // wait for QK to be ready
      ttng.wait_barrier %45, %arg15 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      // load QK
      %result_7, %token_8 = ttng.tmem_load %43[] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64> -> tensor<256x64xf32, #blocked>
      ttng.arrive_barrier %44, 1 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %47 = arith.addi %arg14, %c1_i32 : i32
      %48 = arith.xori %arg15, %c1_i32 : i32
      %49 = arith.cmpi eq, %47, %c2_i32 : i32
      %50 = arith.select %49, %c0_i32, %47 : i32
      %51 = arith.select %49, %48, %arg15 : i32
      %52 = "compute_row_max"(%result_7, %arg3) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %53 = "sub_row_max"(%result_7, %52, %arg3) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %54 = math.exp2 %53 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256x64xf32, #blocked>
      %55 = arith.subf %arg8, %52 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %56 = math.exp2 %55 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %57 = "tt.reduce"(%54) <{axis = 1 : i32}> ({
      ^bb0(%arg20: f32, %arg21: f32):
        %100 = arith.addf %arg20, %arg21 : f32
        tt.reduce.return %100 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %58 = arith.mulf %arg7, %56 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %59 = arith.addf %58, %57 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %60 = tt.addptr %arg5, %arg6 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : !tt.ptr<f32>, i32
      %61 = tt.load %60 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : !tt.ptr<f32>
      %62 = tt.splat %61 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : f32 -> tensor<256x64xf32, #blocked>
      %63 = arith.truncf %54 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      // wait for P to be empty
      ttng.wait_barrier %31, %arg19 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      // store P
      ttng.tmem_store %63, %result_4, %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>
      // signal P ready
      ttng.arrive_barrier %33, 1 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

      // [[CORRECTION]]
      // rematerialized exp2(m_i - m_ij)
      %64 = arith.subf %arg8, %52 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %65 = math.exp2 %64 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %66 = tt.expand_dims %65 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %67 = tt.broadcast %66 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %68 = ttg.memdesc_subview %result_2[%arg16, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %69 = ttg.memdesc_subview %29[%arg16] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      // wait for PV to be ready
      ttng.wait_barrier %69, %arg17 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      // load PV
      %result_9, %token_10 = ttng.tmem_load %68[] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
      %70 = arith.addi %arg16, %c1_i32 : i32
      %71 = arith.xori %arg17, %c1_i32 : i32
      %72 = arith.cmpi eq, %70, %c1_i32 : i32
      %73 = arith.select %72, %c0_i32, %70 : i32
      %74 = arith.select %72, %71, %arg17 : i32
      // o *= alpha
      %75 = arith.mulf %result_9, %67 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked>
      %76 = arith.addf %75, %62 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked>
      %77 = ttg.memdesc_subview %result_2[%73, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // store PV
      %78 = ttng.tmem_store %76, %77[], %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %79 = ttg.memdesc_subview %27[%73] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      // signal PV ready
      ttng.arrive_barrier %79, 1 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

      // [[LOAD V]]
      // wait for V buffer to be consumed
      %80 = ttg.memdesc_subview %13[%arg12] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %80, %arg13 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %81 = ttg.memdesc_subview %17[%arg12] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.barrier_expect %81, 8192 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32}, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %82 = ttg.memdesc_subview %12[%arg12, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
      // issue async load V
      ttng.async_tma_copy_global_to_local %arg2[%arg6, %c0_i32] %82, %81, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>

      // wait for V to be ready
      ttng.wait_barrier %81, %arg13 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      // wait for P to be ready
      ttng.wait_barrier %79, %74, %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %83 = ttg.memdesc_subview %29[%73] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      // wait for PV to be ready
      ttng.wait_barrier %33, %arg19 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      // issue async MMA PV
      %84 = ttng.tc_gen5_mma %result_4, %82, %77[], %true, %true, %80[%true], %83[%true], %31[%true] {loop.cluster = 0 : i32, loop.stage = 4 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>

      // phase increments
      %85 = arith.addi %arg10, %c1_i32 : i32
      %86 = arith.xori %arg11, %c1_i32 : i32
      %87 = arith.cmpi eq, %85, %c3_i32 : i32
      %88 = arith.select %87, %c0_i32, %85 : i32
      %89 = arith.select %87, %86, %arg11 : i32
      %90 = arith.addi %arg12, %c1_i32 : i32
      %91 = arith.xori %arg13, %c1_i32 : i32
      %92 = arith.cmpi eq, %90, %c3_i32 : i32
      %93 = arith.select %92, %c0_i32, %90 : i32
      %94 = arith.select %92, %91, %arg13 : i32
      %95 = arith.addi %arg18, %c1_i32 : i32
      %96 = arith.xori %arg19, %c1_i32 : i32
      %97 = arith.cmpi eq, %95, %c1_i32 : i32
      %98 = arith.select %97, %c0_i32, %95 : i32
      %99 = arith.select %97, %96, %arg19 : i32
      scf.yield %59, %52, %0, %88, %89, %93, %94, %50, %51, %73, %74, %98, %99 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
    } {tt.scheduled_max_stage = 4 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32]}
    // wait for final QK to complete (technically not needed)
    %36 = ttg.memdesc_subview %29[%35#9] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %36, %35#10 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    ttng.inval_barrier %34 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %33 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %31 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %30 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %29 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %28 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %27 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %37 = ttg.memdesc_subview %result_2[%35#9, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %38 = ttg.memdesc_subview %24[%35#7] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>

    // wait for final PV to complete (mandatory)
    ttng.wait_barrier %38, %35#8 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
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

    // load final PV
    %result_5, %token_6 = ttng.tmem_load %37[%35#2] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
    "use"(%35#0, %result_5, %35#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}

