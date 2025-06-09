//
// After tritongpu-rewrite-partition-dependencies
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
    %c-1_i32 = arith.constant -1 : i32
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
    %result_4 = ttng.tmem_alloc {loop.cluster = 0 : i32, loop.stage = 4 : i32} : () -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>
    %31 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %32 = ttg.memdesc_subview %31[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %32, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %33 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %34 = ttg.memdesc_subview %33[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %34, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %31, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    // Triple buffer for m_i
    // [0, 1, 2]
    //
    // 0: stores the initial value of m_i for the first (and only) multiplicity
    //    branch. technically not needed if there is only 1 multiplicity branch
    //
    // 1, 2: steady-state double-buffer for m_i since both can be live at the
    //       same time across different partitions. technically if the compiler
    //       knew the access order of m_i it could reduce this to 1
    %35 = ttg.local_alloc : () -> !ttg.memdesc<3x256xf32, #shared1, #smem, mutable>
    %36 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %37 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    // store initial value of m_i
    // this is m_i(-1)
    %38 = ttg.memdesc_subview %35[%c0_i32, %c0_i32] : !ttg.memdesc<3x256xf32, #shared1, #smem, mutable> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256>
    ttg.local_store %cst, %38 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256>

    // initialize barrier pairs 0 as ready
    %39 = ttg.memdesc_subview %36[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %40 = ttg.memdesc_subview %37[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %39, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %40, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %39, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>

    // initialize barrier pairs 1 and 2 as empty
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

    %45:19 = scf.for %arg6 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg7 = %cst_1, %arg8 = %cst, %arg9 = %2, %arg10 = %c0_i32, %arg11 = %c0_i32, %arg12 = %c0_i32, %arg13 = %c0_i32, %arg14 = %c0_i32, %arg15 = %c0_i32, %arg16 = %c0_i32, %arg17 = %c0_i32, %arg18 = %c0_i32, %arg19 = %c0_i32, %arg20 = %c0_i32, %arg21 = %c0_i32, %arg22 = %c-1_i32, %arg23 = %c0_i32, %arg24 = %c0_i32, %arg25 = %c0_i32) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
      %49 = ttg.memdesc_subview %4[%arg10] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %49, %arg11 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %50 = ttg.memdesc_subview %8[%arg10] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.barrier_expect %50, 8192 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32}, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %51 = ttg.memdesc_subview %3[%arg10, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
      ttng.async_tma_copy_global_to_local %arg1[%arg6, %c0_i32] %51, %50, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
      %52 = ttg.memdesc_trans %51 {loop.cluster = 2 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable, 3x64x64>
      ttng.wait_barrier %50, %arg11 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %53 = ttg.memdesc_subview %result[%arg14, %c0_i32, %c0_i32] : !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %54 = ttg.memdesc_subview %24[%arg14] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %54, %arg15, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %55 = ttg.memdesc_subview %21[%arg14] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %56 = ttng.tc_gen5_mma %arg0, %52, %53[], %false, %true, %49[%true], %55[%true] {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared2, #smem, mutable, 3x64x64>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %55, %arg15 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %result_7, %token_8 = ttng.tmem_load %53[] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64> -> tensor<256x64xf32, #blocked>
      ttng.arrive_barrier %54, 1 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %57 = arith.addi %arg14, %c1_i32 : i32
      %58 = arith.xori %arg15, %c1_i32 : i32
      %59 = arith.cmpi eq, %57, %c2_i32 : i32
      %60 = arith.select %59, %c0_i32, %57 : i32
      %61 = arith.select %59, %58, %arg15 : i32
      %62 = "compute_row_max"(%result_7, %arg3) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %63 = arith.addi %arg24, %c1_i32 : i32
      %64 = arith.xori %arg25, %c1_i32 : i32
      %65 = arith.cmpi eq, %63, %c3_i32 : i32
      %66 = arith.select %65, %64, %arg25 : i32
      %67 = arith.select %65, %c1_i32, %63 : i32

      // wait for m_i(i) to be empty
      %68 = ttg.memdesc_subview %35[%67, %c0_i32] : !ttg.memdesc<3x256xf32, #shared1, #smem, mutable> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256>
      %69 = ttg.memdesc_subview %36[%67] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %70 = ttg.memdesc_subview %37[%67] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %70, %66 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      // store m_i(i)
      ttg.local_store %62, %68 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256>
      // signal m_i(i) is ready
      ttng.arrive_barrier %69, 1 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>

      %71 = "sub_row_max"(%result_7, %62, %arg3) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %72 = math.exp2 %71 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256x64xf32, #blocked>
      %73 = arith.subf %arg8, %62 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %74 = math.exp2 %73 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %75 = "tt.reduce"(%72) <{axis = 1 : i32}> ({
      ^bb0(%arg26: f32, %arg27: f32):
        %136 = arith.addf %arg26, %arg27 : f32
        tt.reduce.return %136 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %76 = arith.mulf %arg7, %74 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %77 = arith.addf %76, %75 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %78 = tt.addptr %arg5, %arg6 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : !tt.ptr<f32>, i32
      %79 = tt.load %78 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : !tt.ptr<f32>
      %80 = tt.splat %79 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : f32 -> tensor<256x64xf32, #blocked>
      %81 = arith.truncf %72 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      ttng.wait_barrier %31, %arg19 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tmem_store %81, %result_4, %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>
      ttng.arrive_barrier %33, 1 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %82 = arith.addi %arg20, %c1_i32 : i32
      %83 = arith.xori %arg21, %c1_i32 : i32
      %84 = arith.cmpi eq, %82, %c3_i32 : i32
      %85 = arith.select %84, %83, %arg21 : i32
      %86 = arith.select %84, %c1_i32, %82 : i32

      // wait for m_i(i-1) to be ready
      %87 = ttg.memdesc_subview %35[%86, %c0_i32] : !ttg.memdesc<3x256xf32, #shared1, #smem, mutable> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256>
      %88 = ttg.memdesc_subview %36[%86] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %89 = ttg.memdesc_subview %37[%86] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %88, %85 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      // load m_i(i-1)
      %90 = ttg.local_load %87 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256> -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      // signal m_i(i-1) is empty
      ttng.arrive_barrier %89, 1 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %91 = arith.addi %arg22, %c1_i32 : i32
      %92 = arith.xori %arg23, %c1_i32 : i32
      %93 = arith.cmpi eq, %91, %c3_i32 : i32
      %94 = arith.select %93, %92, %arg23 : i32
      %95 = arith.select %93, %c1_i32, %91 : i32

      // wait for m_i(i) to be ready
      %96 = ttg.memdesc_subview %35[%95, %c0_i32] : !ttg.memdesc<3x256xf32, #shared1, #smem, mutable> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256>
      %97 = ttg.memdesc_subview %36[%95] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %98 = ttg.memdesc_subview %37[%95] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %97, %94 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      // load m_i(i)
      %99 = ttg.local_load %96 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<256xf32, #shared1, #smem, mutable, 3x256> -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      // signal m_i(i) is empty
      ttng.arrive_barrier %98, 1 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>

      %100 = arith.subf %99, %90 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %101 = math.exp2 %100 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %102 = tt.expand_dims %101 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %103 = tt.broadcast %102 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %104 = ttg.memdesc_subview %result_2[%arg16, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %105 = ttg.memdesc_subview %29[%arg16] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %105, %arg17 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %result_9, %token_10 = ttng.tmem_load %104[] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
      %106 = arith.addi %arg16, %c1_i32 : i32
      %107 = arith.xori %arg17, %c1_i32 : i32
      %108 = arith.cmpi eq, %106, %c1_i32 : i32
      %109 = arith.select %108, %c0_i32, %106 : i32
      %110 = arith.select %108, %107, %arg17 : i32
      %111 = arith.mulf %result_9, %103 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked>
      %112 = arith.addf %111, %80 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked>
      %113 = ttg.memdesc_subview %result_2[%109, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %114 = ttng.tmem_store %112, %113[], %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %115 = ttg.memdesc_subview %27[%109] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.arrive_barrier %115, 1 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %116 = ttg.memdesc_subview %13[%arg12] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %116, %arg13 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %117 = ttg.memdesc_subview %17[%arg12] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.barrier_expect %117, 8192 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32}, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %118 = ttg.memdesc_subview %12[%arg12, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
      ttng.async_tma_copy_global_to_local %arg2[%arg6, %c0_i32] %118, %117, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>
      ttng.wait_barrier %117, %arg13 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %115, %110, %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %119 = ttg.memdesc_subview %29[%109] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %33, %arg19 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %120 = ttng.tc_gen5_mma %result_4, %118, %113[], %true, %true, %116[%true], %119[%true], %31[%true] {loop.cluster = 0 : i32, loop.stage = 4 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable, 3x64x64>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %121 = arith.addi %arg10, %c1_i32 : i32
      %122 = arith.xori %arg11, %c1_i32 : i32
      %123 = arith.cmpi eq, %121, %c3_i32 : i32
      %124 = arith.select %123, %c0_i32, %121 : i32
      %125 = arith.select %123, %122, %arg11 : i32
      %126 = arith.addi %arg12, %c1_i32 : i32
      %127 = arith.xori %arg13, %c1_i32 : i32
      %128 = arith.cmpi eq, %126, %c3_i32 : i32
      %129 = arith.select %128, %c0_i32, %126 : i32
      %130 = arith.select %128, %127, %arg13 : i32
      %131 = arith.addi %arg18, %c1_i32 : i32
      %132 = arith.xori %arg19, %c1_i32 : i32
      %133 = arith.cmpi eq, %131, %c1_i32 : i32
      %134 = arith.select %133, %c0_i32, %131 : i32
      %135 = arith.select %133, %132, %arg19 : i32
      scf.yield %77, %62, %0, %124, %125, %129, %130, %60, %61, %109, %110, %134, %135, %86, %85, %95, %94, %67, %66 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
    } {tt.scheduled_max_stage = 4 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32]}
    ttg.local_dealloc %35 : !ttg.memdesc<3x256xf32, #shared1, #smem, mutable>
    ttng.inval_barrier %39 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %40 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %41 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %42 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %43 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %44 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %36 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %37 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %46 = ttg.memdesc_subview %29[%45#9] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %46, %45#10 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %34 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %33 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %31 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %30 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %29 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %28 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %27 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %47 = ttg.memdesc_subview %result_2[%45#9, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %48 = ttg.memdesc_subview %24[%45#7] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.wait_barrier %48, %45#8 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
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
    %result_5, %token_6 = ttng.tmem_load %47[%45#2] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
    "use"(%45#0, %result_5, %45#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}

