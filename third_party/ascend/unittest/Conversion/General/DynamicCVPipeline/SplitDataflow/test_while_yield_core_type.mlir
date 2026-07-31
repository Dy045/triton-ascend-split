// RUN: split-file %s %t
// RUN: triton-opt --data-dependency-analysis %t/valid.mlir | FileCheck %s --check-prefix=VALID
// RUN: triton-opt --data-dependency-analysis %t/mismatch.mlir | FileCheck %s --check-prefix=MISMATCH

//--- valid.mlir

// VALID-LABEL: func.func @while_yield_core_type_match
// VALID: scf.while
// VALID-NOT: triton_ascend.dynamic_cv_pipeline.rc
module {
  func.func @while_yield_core_type_match(%init: i32) {
    %result = scf.while (%arg = %init) : (i32) -> i32 {
      %condition = arith.constant true
      scf.condition(%condition) %arg : i32
    } do {
    ^bb0(%arg: i32):
      %inner = scf.while (%inner_arg = %arg) : (i32) -> i32 {
        %condition = arith.constant true
        scf.condition(%condition) %inner_arg : i32
      } do {
      ^bb0(%inner_arg: i32):
        scf.yield {ssbuffer.core_type = "VECTOR"} %inner_arg : i32
      } attributes {ssbuffer.core_type = "VECTOR"}
      scf.yield {ssbuffer.core_type = "VECTOR"} %inner : i32
    } attributes {ssbuffer.core_type = "VECTOR"}
    func.return
  }
}

//--- mismatch.mlir

// MISMATCH: triton_ascend.dynamic_cv_pipeline.rc = 1 : i32
module {
  func.func @while_yield_core_type_mismatch(%init: i32) {
    %result = scf.while (%arg = %init) : (i32) -> i32 {
      %condition = arith.constant true
      scf.condition(%condition) %arg : i32
    } do {
    ^bb0(%arg: i32):
      %inner = scf.while (%inner_arg = %arg) : (i32) -> i32 {
        %condition = arith.constant true
        scf.condition(%condition) %inner_arg : i32
      } do {
      ^bb0(%inner_arg: i32):
        scf.yield {ssbuffer.core_type = "CUBE"} %inner_arg : i32
      } attributes {ssbuffer.core_type = "CUBE"}
      scf.yield {ssbuffer.core_type = "VECTOR"} %inner : i32
    } attributes {ssbuffer.core_type = "VECTOR"}
    func.return
  }
}
