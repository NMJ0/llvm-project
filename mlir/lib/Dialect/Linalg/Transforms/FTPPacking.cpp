//===- FTPPacking.cpp - Fully Temporal-Parallel matmul packing pass -------===//
//
// This file implements the FTPPacking pass, which detects groups of
// linalg.matmul operations that share the same weight (RHS) operand across
// multiple SNN timesteps and fuses them into a single batched matmul.
//
// Motivation (LoAS paper, Algorithm 1):
//   Standard SNN unrolled loop produces T sequential matmuls:
//     for t in [0,T):  O[t] = spike[t] @ W    (then bias, accumulate)
//   FTP insight: place t-dim at innermost of IP dataflow, parallelize:
//     O_all = stack(spike[0..T-1]) @ W         (one (T,K)@(K,N) matmul)
//     result = reduce_sum(bias(O_all), dim=0)
//
// Detection heuristic (generic SNN, any T, any layer size):
//   1. Group all linalg.matmul ops by their RHS (weight) Value.
//   2. Retain groups of size >= 2 where each LHS is a spike tensor,
//      identified by tracing: LHS <- linalg.generic(uitofp) <- linalg.generic
//      (cmpf oge, producing i1).  This is the standard MLIR lowering of the
//      LIF threshold comparison.
//   3. For each such group, emit the FTP replacement and erase the originals.
//
// Transformation:
//   Given group [mm_0, mm_1, ..., mm_{T-1}] sharing RHS=W:
//   a) Stack LHS:  stacked = insert_slice(spike[0..T-1]) : (T,K)
//   b) Batch matmul: out_all = linalg.matmul(stacked, W) : (T,N)
//   c) Bias broadcast (optional): if all mm_i share same bias B,
//      biased = linalg.generic(out_all + B) : (T,N)
//   d) Reduce-sum: reduced = linalg.generic(biased, reduction over dim 0)
//      : (1,N)
//   e) Replace uses of the final accumulate-chain result with `reduced`.
//   f) Erase: accumulate generics, bias-add generics, original matmuls
//      (in that order, to satisfy SSA def-use constraints).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

// Required by Passes.h.inc: affine::AffineDialect is referenced in the
// generated getDependentDialects() body.
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ftp-packing"

namespace mlir {
namespace linalg {

#define GEN_PASS_DEF_FTPPACKING
#include "mlir/Dialect/Linalg/Passes.h.inc"

} // namespace linalg
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

namespace {

/// Return true when \p op is a linalg.generic that implements a pointwise
/// binary addition over its two inputs and one output.
///
/// The pattern covers both the bias-add and the timestep accumulate step:
///   iterator_types = all "parallel"
///   body           = { %r = arith.addf %in0, %in1 ; linalg.yield %r }
/// The indexing maps are intentionally not checked here because they differ
/// between the bias-add (#map / #map1 / #map) and the accumulate (#map * 3).
static bool isAddGeneric(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
    return false;
  // Every iterator must be parallel.
  for (Attribute attr : op.getIteratorTypes())
    if (cast<linalg::IteratorTypeAttr>(attr).getValue() !=
        utils::IteratorType::parallel)
      return false;
  // Body: exactly two ops — arith.addf + linalg.yield.
  Block &body = op.getRegion().front();
  if (body.getOperations().size() != 2)
    return false;
  auto addOp = dyn_cast<arith::AddFOp>(&body.front());
  if (!addOp)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.back());
  if (!yieldOp || yieldOp.getValues().size() != 1 ||
      yieldOp.getValues()[0] != addOp.getResult())
    return false;
  return true;
}

/// Trace back through the def-use chain to determine whether \p val is a
/// spike tensor — i.e., the output of a linalg.generic that converts i1->f32
/// via uitofp (or extui for some lowerings).
///
/// Standard LIF lowering pattern in MLIR:
///   %cmp   = linalg.generic { cmpf oge } ins(%membrane) -> tensor<...xi1>
///   %spike = linalg.generic { uitofp   } ins(%cmp)      -> tensor<...xf32>
///
/// The function also accepts an optional mulf scaling layer on top (some
/// lowerings compute spike * vth as a separate generic before the soft-reset
/// subtraction — but the matmul LHS is always the unscaled spike_f).
static bool isSpikeValue(Value val) {
  auto *defOp = val.getDefiningOp();
  if (!defOp)
    return false;

  auto gen = dyn_cast<linalg::GenericOp>(defOp);
  if (!gen)
    return false;

  Block &body = gen.getRegion().front();
  if (body.getOperations().size() != 2)
    return false;

  // Primary case: i1 -> f32 via uitofp.
  if (isa<arith::UIToFPOp>(body.front()))
    return true;

  // Alternative: i1 -> integer via extui (some pipeline variants).
  if (isa<arith::ExtUIOp>(body.front()))
    return true;

  // Secondary case: allow a mulf scaling layer whose first input is itself
  // a spike (handles rare lowerings that scale before passing to matmul).
  if (isa<arith::MulFOp>(body.front()) && gen.getNumDpsInputs() >= 1)
    return isSpikeValue(gen.getDpsInputs()[0]);

  return false;
}

/// If the first user of \p matmulResult that is an add-generic also has a
/// 1-D second input, that second input is the bias vector.  Returns
/// {true, biasValue} on success, {false, {}} otherwise.
static std::pair<bool, Value> detectBiasAdd(Value matmulResult) {
  for (Operation *user : matmulResult.getUsers()) {
    auto gen = dyn_cast<linalg::GenericOp>(user);
    if (!gen || !isAddGeneric(gen))
      continue;
    // One input must be matmulResult; the other is the candidate bias.
    Value other;
    for (Value inp : gen.getDpsInputs())
      if (inp != matmulResult)
        other = inp;
    if (!other)
      continue;
    // Bias is 1-D, matmul output is 2-D.
    auto otherType = dyn_cast<RankedTensorType>(other.getType());
    auto selfType  = dyn_cast<RankedTensorType>(matmulResult.getType());
    if (otherType && selfType &&
        otherType.getRank() == 1 && selfType.getRank() == 2)
      return {true, other};
  }
  return {false, Value{}};
}

/// Walk the accumulate chain that chains together per-timestep biased outputs:
///
///   acc[0] = addf(zero_init,  biased[0])
///   acc[1] = addf(acc[0],     biased[1])
///   ...
///   acc[T-1] = addf(acc[T-2], biased[T-1])   <- returned value
///
/// \p biasedOutputs must be in program order (timestep 0 first).
/// On success, \p accumOps is filled in the same order and the function
/// returns acc[T-1].  Returns a null Value if the chain cannot be found.
static Value
findFinalAccumulate(ArrayRef<Value> biasedOutputs,
                    SmallVectorImpl<linalg::GenericOp> &accumOps) {
  Value curAcc; // null until first accumulate op is found
  for (Value biased : biasedOutputs) {
    bool found = false;
    for (Operation *user : biased.getUsers()) {
      auto gen = dyn_cast<linalg::GenericOp>(user);
      if (!gen || !isAddGeneric(gen))
        continue;
      // gen must consume `biased` as one of its inputs.
      bool usesBiased = false;
      for (Value inp : gen.getDpsInputs())
        if (inp == biased)
          usesBiased = true;
      if (!usesBiased)
        continue;
      // If we already have a previous accumulator it must be the other input.
      if (curAcc) {
        bool usesCurAcc = false;
        for (Value inp : gen.getDpsInputs())
          if (inp == curAcc)
            usesCurAcc = true;
        if (!usesCurAcc)
          continue;
      }
      accumOps.push_back(gen);
      curAcc = gen.getResult(0);
      found  = true;
      break;
    }
    if (!found)
      return {}; // chain broken — bail out
  }
  return curAcc;
}

//===----------------------------------------------------------------------===//
// MatmulGroup: T matmuls that share the same weight (RHS)
//===----------------------------------------------------------------------===//

struct MatmulGroup {
  Value sharedRHS;                         ///< Common weight across timesteps.
  SmallVector<linalg::MatmulOp> matmuls;   ///< Matmul ops, program order.
  SmallVector<Value>            lhsSpikes; ///< Spike LHS per timestep.
  bool  hasBias = false;
  Value bias;                              ///< Shared bias vector (if hasBias).
  SmallVector<linalg::GenericOp> biasOps; ///< Bias-add after each matmul.
  SmallVector<linalg::GenericOp> accumOps;///< Timestep accumulate chain.
  Value finalAcc;                          ///< Result of last accumulate op.

  int64_t T() const { return static_cast<int64_t>(matmuls.size()); }
  int64_t K() const {
    return cast<RankedTensorType>(lhsSpikes[0].getType()).getDimSize(1);
  }
  int64_t N() const {
    return cast<RankedTensorType>(sharedRHS.getType()).getDimSize(1);
  }
};

//===----------------------------------------------------------------------===//
// collectMatmulGroups
//===----------------------------------------------------------------------===//

/// Walk the function and return one MatmulGroup per distinct weight Value that
/// is used as the RHS of >= 2 linalg.matmul ops whose LHS values are spikes.
///
/// llvm::MapVector preserves insertion order so that groups are processed in
/// program order — important for determinism in multi-layer SNNs.
static SmallVector<MatmulGroup, 4>
collectMatmulGroups(func::FuncOp funcOp) {
  // MapVector: insertion order == walk (program) order for the first matmul
  // encountered with each weight.  Subsequent matmuls for the same weight are
  // appended to the vector in walk order, so per-group order is also correct.
  llvm::MapVector<Value, SmallVector<linalg::MatmulOp>> byRHS;
  funcOp.walk([&](linalg::MatmulOp mm) {
    byRHS[mm.getInputs()[1]].push_back(mm);
  });

  SmallVector<MatmulGroup, 4> groups;
  for (auto &[rhs, mms] : byRHS) {
    if (mms.size() < 2)
      continue;

    // All LHS values must trace back to spike tensors.
    SmallVector<Value> spikes;
    spikes.reserve(mms.size());
    bool allSpike = true;
    for (auto mm : mms) {
      Value lhs = mm.getInputs()[0];
      if (!isSpikeValue(lhs)) { allSpike = false; break; }
      spikes.push_back(lhs);
    }
    if (!allSpike)
      continue;

    // All output tensors must share the same (1, N) type.
    auto outType = cast<RankedTensorType>(mms[0].getResult(0).getType());
    bool sameShape = true;
    for (auto mm : mms)
      if (cast<RankedTensorType>(mm.getResult(0).getType()) != outType) {
        sameShape = false; break;
      }
    if (!sameShape)
      continue;

    MatmulGroup grp;
    grp.sharedRHS = rhs;
    grp.matmuls   = mms;
    grp.lhsSpikes = spikes;

    // Detect bias from the first matmul's immediate user.
    auto [hasBias, biasVal] = detectBiasAdd(mms[0].getResult(0));
    grp.hasBias = hasBias;
    grp.bias    = biasVal;

    // Build the ordered list of "biased outputs" (i.e. the value after the
    // optional bias-add for each timestep) and collect bias-add ops.
    SmallVector<Value> biasedOutputs;
    biasedOutputs.reserve(mms.size());
    for (auto mm : mms) {
      Value afterBias = mm.getResult(0);
      if (hasBias) {
        for (Operation *user : mm.getResult(0).getUsers()) {
          auto gen = dyn_cast<linalg::GenericOp>(user);
          if (!gen || !isAddGeneric(gen))
            continue;
          // gen must use this matmul result as an input.
          bool usesMM = llvm::any_of(gen.getDpsInputs(),
              [&](Value v){ return v == mm.getResult(0); });
          if (!usesMM)
            continue;
          // One input must be 1-D (the bias).
          bool has1D = llvm::any_of(gen.getDpsInputs(), [&](Value v) {
            return v != mm.getResult(0) &&
                   cast<RankedTensorType>(v.getType()).getRank() == 1;
          });
          if (!has1D)
            continue;
          grp.biasOps.push_back(gen);
          afterBias = gen.getResult(0);
          break;
        }
      }
      biasedOutputs.push_back(afterBias);
    }

    grp.finalAcc = findFinalAccumulate(biasedOutputs, grp.accumOps);
    if (!grp.finalAcc)
      continue; // accumulate chain not found — skip this group

    groups.push_back(std::move(grp));
  }
  return groups;
}

//===----------------------------------------------------------------------===//
// emitFTPReplacement
//===----------------------------------------------------------------------===//

/// Emit the FTP-packed replacement ops for \p grp, inserting before the
/// first matmul in the group.  Returns the (1,N) Value that semantically
/// replaces \p grp.finalAcc (i.e. the reduce-sum result).
static Value emitFTPReplacement(MatmulGroup &grp, OpBuilder &builder,
                                Location loc) {
  MLIRContext *ctx = builder.getContext();
  const int64_t T = grp.T();
  const int64_t K = grp.K();
  const int64_t N = grp.N();

  // ── Step 1: stack T spike tensors (1,K) into one (T,K) tensor ──────────
  // We thread the result of each insert_slice through as the destination of
  // the next, so every row is written exactly once before the matmul reads it.
  Value stacked = tensor::EmptyOp::create(builder,
      loc, ArrayRef<int64_t>{T, K}, builder.getF32Type());

  for (int64_t t = 0; t < T; ++t) {
    SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(t),
                                         builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes   = {builder.getIndexAttr(1),
                                         builder.getIndexAttr(K)};
    SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                         builder.getIndexAttr(1)};
    stacked = tensor::InsertSliceOp::create(builder,
        loc, grp.lhsSpikes[t], stacked, offsets, sizes, strides);
  }

  // ── Step 2: single (T,K) @ (K,N) → (T,N) matmul ───────────────────────
  auto ftpOutType = RankedTensorType::get({T, N}, builder.getF32Type());
  Value zero = arith::ConstantOp::create(builder,
      loc, builder.getF32Type(), builder.getF32FloatAttr(0.0f));
  Value ftpEmpty = tensor::EmptyOp::create(builder,
      loc, ArrayRef<int64_t>{T, N}, builder.getF32Type());
  Value ftpZero = linalg::FillOp::create(builder,
      loc, ValueRange{zero}, ValueRange{ftpEmpty}).getResult(0);

  Value ftpMatmul = linalg::MatmulOp::create(builder,
      loc, TypeRange{ftpOutType},
      ValueRange{stacked, grp.sharedRHS},
      ValueRange{ftpZero}).getResult(0);

  // ── Step 3 (optional): broadcast bias (N,) over T rows → (T,N) ─────────
  //   Affine maps:
  //     in0  : (d0, d1) → (d0, d1)   identity for (T,N)
  //     in1  : (d0, d1) → (d1)       broadcast bias over the t-dim
  //     out  : (d0, d1) → (d0, d1)
  Value afterBias = ftpMatmul;
  if (grp.hasBias && grp.bias) {
    AffineMap identMap = AffineMap::getMultiDimIdentityMap(2, ctx);
    AffineMap biasMap =
        AffineMap::get(2, 0,
                       SmallVector<AffineExpr>{builder.getAffineDimExpr(1)},
                       ctx);

    Value biasOut = tensor::EmptyOp::create(builder,
        loc, ArrayRef<int64_t>{T, N}, builder.getF32Type());

    afterBias = linalg::GenericOp::create(builder,
        loc, TypeRange{ftpOutType},
        ValueRange{ftpMatmul, grp.bias},
        ValueRange{biasOut},
        ArrayRef<AffineMap>{identMap, biasMap, identMap},
        ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                      utils::IteratorType::parallel},
        [&](OpBuilder &b, Location l, ValueRange args) {
          linalg::YieldOp::create(b, l,
              arith::AddFOp::create(b, l, args[0], args[1]).getResult());
        }).getResult(0);
  }

  // ── Step 4: reduce-sum over t-dim: (T,N) → (1,N) ───────────────────────
  //   Affine maps:
  //     in  : (d0, d1) → (d0, d1)         identity over (T,N)
  //     out : (d0, d1) → ( 0, d1)         constant 0 on d0 collapses t-dim
  //   Iterator types: [reduction, parallel]
  AffineExpr d1 = builder.getAffineDimExpr(1);
  AffineMap inMap = AffineMap::getMultiDimIdentityMap(2, ctx);
  AffineMap outMap =
      AffineMap::get(2, 0,
                     SmallVector<AffineExpr>{
                         builder.getAffineConstantExpr(0), d1},
                     ctx);

  auto reducedType = RankedTensorType::get({1, N}, builder.getF32Type());
  Value reduceEmpty = tensor::EmptyOp::create(builder,
      loc, ArrayRef<int64_t>{1, N}, builder.getF32Type());
  Value reduceZero = linalg::FillOp::create(builder,
      loc, ValueRange{zero}, ValueRange{reduceEmpty}).getResult(0);

  Value reduced = linalg::GenericOp::create(builder,
      loc, TypeRange{reducedType},
      ValueRange{afterBias},
      ValueRange{reduceZero},
      ArrayRef<AffineMap>{inMap, outMap},
      ArrayRef<utils::IteratorType>{utils::IteratorType::reduction,
                                    utils::IteratorType::parallel},
      [&](OpBuilder &b, Location l, ValueRange args) {
        // args[0] = element of (T,N) input
        // args[1] = current partial sum in (1,N) output
        linalg::YieldOp::create(b, l,
            arith::AddFOp::create(b, l, args[0], args[1]).getResult());
      }).getResult(0);

  return reduced;
}

//===----------------------------------------------------------------------===//
// The pass — runOnOperation
//===----------------------------------------------------------------------===//

struct FTPPackingPass
    : public mlir::linalg::impl::FTPPackingBase<FTPPackingPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // ── 1. Collect spike-matmul groups ────────────────────────────────────
    SmallVector<MatmulGroup, 4> groups = collectMatmulGroups(funcOp);

    if (groups.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[ftp-packing] no groups found in '"
                 << funcOp.getName() << "'\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[ftp-packing] " << groups.size()
               << " group(s) in '" << funcOp.getName() << "'\n");

    OpBuilder builder(funcOp.getContext());

    // ── 2. Process each group ─────────────────────────────────────────────
    for (MatmulGroup &grp : groups) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[ftp-packing]   T=" << grp.T()
                 << " K=" << grp.K() << " N=" << grp.N()
                 << " hasBias=" << grp.hasBias << '\n');

      // ── Find the correct insertion point ────────────────────────────────
      // We must insert AFTER the last op that defines any value we consume:
      // all spike LHS tensors, the shared RHS weight, and the bias (if any).
      // Inserting before the first matmul is wrong when later spikes are
      // defined after mm_0 — those values would not yet dominate the use.
      //
      // Strategy: collect every defining operation for values we need, then
      // pick the one that appears latest in the block (i.e. has the highest
      // index when iterating forward).  Insert immediately after that op.
      //
      // All ops in a group must be in the same block (they share a function
      // body), so block-order comparison via iterators is valid.
      Block *parentBlock = grp.matmuls[0].getOperation()->getBlock();

      // Gather all values the replacement will directly consume.
      SmallVector<Value> neededValues(grp.lhsSpikes.begin(),
                                      grp.lhsSpikes.end());
      neededValues.push_back(grp.sharedRHS);
      if (grp.hasBias && grp.bias)
        neededValues.push_back(grp.bias);

      // Walk the block once to assign each op a position index, then find
      // the maximum index among the defining ops of our needed values.
      // Block::getOperations() returns ops in program order.
      llvm::DenseMap<Operation *, unsigned> opIndex;
      unsigned idx = 0;
      for (Operation &op : parentBlock->getOperations())
        opIndex[&op] = idx++;

      Operation *lastDef = nullptr;
      unsigned   lastIdx = 0;
      for (Value v : neededValues) {
        Operation *defOp = v.getDefiningOp();
        if (!defOp)
          continue; // block argument — always dominates everything
        auto it = opIndex.find(defOp);
        if (it == opIndex.end())
          continue; // defined outside this block — also always dominates
        if (!lastDef || it->second > lastIdx) {
          lastDef = defOp;
          lastIdx = it->second;
        }
      }

      // If every needed value is a block argument or defined outside the
      // block, fall back to inserting before the first matmul (safe because
      // block arguments dominate everything).
      Operation *insertionPt;
      if (lastDef)
        insertionPt = lastDef->getNextNode(); // insert *after* lastDef
      else
        insertionPt = grp.matmuls[0].getOperation();

      builder.setInsertionPoint(insertionPt);
      Location loc = grp.matmuls[0].getOperation()->getLoc();

      // ── 2a. Emit the FTP replacement ops ────────────────────────────────
      Value newResult = emitFTPReplacement(grp, builder, loc);

      // ── 2b. Redirect all downstream uses of the old accumulate result ───
      grp.finalAcc.replaceAllUsesWith(newResult);

      // ── 2c. Erase obsolete ops (SSA requires: users before defs) ────────
      //   Order: accumulate generics → bias-add generics → matmuls.
      //   Within each category we erase in reverse program order so that
      //   each op is erased only after its SSA users have been erased.

      for (auto it = grp.accumOps.rbegin(); it != grp.accumOps.rend(); ++it)
        (*it)->erase();

      for (auto it = grp.biasOps.rbegin(); it != grp.biasOps.rend(); ++it)
        (*it)->erase();

      for (auto it = grp.matmuls.rbegin(); it != grp.matmuls.rend(); ++it)
        (*it)->erase();
    }
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Public pass-creation entry point
//===----------------------------------------------------------------------===//

namespace mlir {
namespace linalg {

std::unique_ptr<Pass> createFTPPackingPass() {
  return std::make_unique<FTPPackingPass>();
}

} // namespace linalg
} // namespace mlir