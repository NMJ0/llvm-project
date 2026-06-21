//===- FTPPacking.cpp - Fully Temporal-Parallel matmul packing pass -------===//
//
// Updated to robustly support arbitrary spatial pooling and flattening
// prior to the FC layer by removing strict `uitofp` origin checks.
// Extended to pack `linalg.conv_2d_nchw_fchw` convolutions over time.
// Extended to pack `linalg.pooling_nchw_sum` poolings over time.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"
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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ftp-packing"

using namespace mlir;
using namespace mlir::linalg;

namespace {

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

/// Sorts the block topologically to resolve any SSA dominance violations
/// introduced by batching operations across unrolled timesteps.
static void sortBlockTopologically(Block *block) {
  SmallVector<Operation*> ops;
  for (auto &op : *block) {
    if (!isa<func::ReturnOp>(op))
      ops.push_back(&op);
  }

  llvm::DenseSet<Operation*> scheduled;
  Operation* insertPoint = nullptr;
  bool changed = true;
 
  while (changed) {
    changed = false;
    for (Operation *op : ops) {
      if (scheduled.count(op)) continue;
     
      bool ready = true;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (def && def->getBlock() == block && !scheduled.count(def)) {
          ready = false; break;
        }
      }
     
      if (ready) {
        if (insertPoint) op->moveAfter(insertPoint);
        else op->moveBefore(block, block->begin());
       
        insertPoint = op;
        scheduled.insert(op);
        changed = true;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Matmul Grouping
//===----------------------------------------------------------------------===//

struct MatmulGroup {
  Value sharedRHS;
  SmallVector<linalg::MatmulOp> matmuls;
  SmallVector<Value> lhsVals;
  bool hasBias = false;
  Value bias;
  SmallVector<linalg::GenericOp> biasOps;

  int64_t T() const { return static_cast<int64_t>(matmuls.size()); }
  int64_t K() const { return cast<RankedTensorType>(lhsVals[0].getType()).getDimSize(1); }
  int64_t N() const { return cast<RankedTensorType>(sharedRHS.getType()).getDimSize(1); }
};

static SmallVector<MatmulGroup, 4> collectMatmulGroups(func::FuncOp funcOp) {
  llvm::MapVector<Value, SmallVector<linalg::MatmulOp>> byRHS;
  funcOp.walk([&](linalg::MatmulOp mm) {
    byRHS[mm.getInputs()[1]].push_back(mm);
  });

  SmallVector<MatmulGroup, 4> groups;
  for (auto &[rhs, mms] : byRHS) {
    if (mms.size() < 2) continue;

    SmallVector<Value> lhsVals;
    bool sameShape = true;
    auto firstLhsTy = cast<RankedTensorType>(mms[0].getInputs()[0].getType());
   
    for (auto mm : mms) {
      Value lhs = mm.getInputs()[0];
      lhsVals.push_back(lhs);
      if (cast<RankedTensorType>(lhs.getType()) != firstLhsTy) {
         sameShape = false;
         break;
      }
    }
    if (!sameShape) continue;

    bool hasSharedBias = true;
    Value sharedBias = nullptr;
    SmallVector<linalg::GenericOp> biasOps;

    for (auto mm : mms) {
      bool foundBias = false;
      for (Operation *user : mm.getResult(0).getUsers()) {
        auto gen = dyn_cast<linalg::GenericOp>(user);
        if (!gen || gen.getNumDpsInputs() != 2) continue;
       
        auto &body = gen.getRegion().front();
        if (!isa<arith::AddFOp>(body.front())) continue;

        Value other = (gen.getDpsInputs()[0] == mm.getResult(0)) ? gen.getDpsInputs()[1] : gen.getDpsInputs()[0];
        auto otherTy = dyn_cast<RankedTensorType>(other.getType());
       
        if (otherTy && otherTy.getRank() == 1) {
           if (!sharedBias) sharedBias = other;
           if (sharedBias == other) {
              foundBias = true;
              biasOps.push_back(gen);
              break;
           }
        }
      }
      if (!foundBias) { hasSharedBias = false; break; }
    }

    MatmulGroup grp;
    grp.sharedRHS = rhs;
    grp.matmuls = mms;
    grp.lhsVals = lhsVals;
    grp.hasBias = hasSharedBias;
    grp.bias = hasSharedBias ? sharedBias : nullptr;
    grp.biasOps = hasSharedBias ? biasOps : SmallVector<linalg::GenericOp>();

    groups.push_back(grp);
  }
  return groups;
}

//===----------------------------------------------------------------------===//
// Convolution Grouping
//===----------------------------------------------------------------------===//

struct ConvGroup {
  Value sharedFilter;
  SmallVector<linalg::Conv2DNchwFchwOp> convs;
  SmallVector<Value> lhsVals;
  SmallVector<Value> outsVals;

  int64_t T() const { return static_cast<int64_t>(convs.size()); }
  RankedTensorType getLhsType() const { return cast<RankedTensorType>(lhsVals[0].getType()); }
  RankedTensorType getOutsType() const { return cast<RankedTensorType>(outsVals[0].getType()); }
};

static SmallVector<ConvGroup, 4> collectConvGroups(func::FuncOp funcOp) {
  llvm::MapVector<Value, SmallVector<linalg::Conv2DNchwFchwOp>> byFilter;
  
  funcOp.walk([&](linalg::Conv2DNchwFchwOp conv) {
    byFilter[conv.getInputs()[1]].push_back(conv);
  });

  SmallVector<ConvGroup, 4> groups;
  for (auto &[filter, convs] : byFilter) {
    if (convs.size() < 2) continue;

    SmallVector<Value> lhsVals;
    SmallVector<Value> outsVals;
    bool sameShape = true;
    
    auto firstLhsTy = cast<RankedTensorType>(convs[0].getInputs()[0].getType());
    auto firstOutsTy = cast<RankedTensorType>(convs[0].getOutputs()[0].getType());

    if (firstLhsTy.getDimSize(0) != 1) continue;

    for (auto conv : convs) {
      Value lhs = conv.getInputs()[0];
      Value outs = conv.getOutputs()[0];
      lhsVals.push_back(lhs);
      outsVals.push_back(outs);

      if (cast<RankedTensorType>(lhs.getType()) != firstLhsTy ||
          cast<RankedTensorType>(outs.getType()) != firstOutsTy) {
        sameShape = false;
        break;
      }
    }
    if (!sameShape) continue;

    ConvGroup grp;
    grp.sharedFilter = filter;
    grp.convs = convs;
    grp.lhsVals = lhsVals;
    grp.outsVals = outsVals;
    groups.push_back(grp);
  }
  return groups;
}

//===----------------------------------------------------------------------===//
// Pooling Grouping
//===----------------------------------------------------------------------===//

struct PoolGroup {
  Value sharedWindow;
  SmallVector<linalg::PoolingNchwSumOp> pools;
  SmallVector<Value> lhsVals;
  SmallVector<Value> outsVals;

  int64_t T() const { return static_cast<int64_t>(pools.size()); }
  RankedTensorType getLhsType() const { return cast<RankedTensorType>(lhsVals[0].getType()); }
  RankedTensorType getOutsType() const { return cast<RankedTensorType>(outsVals[0].getType()); }
};

static SmallVector<PoolGroup, 4> collectPoolGroups(func::FuncOp funcOp) {
  llvm::MapVector<Value, SmallVector<linalg::PoolingNchwSumOp>> byWindow;
  
  funcOp.walk([&](linalg::PoolingNchwSumOp pool) {
    // Input 1 is the pooling window
    byWindow[pool.getInputs()[1]].push_back(pool);
  });

  SmallVector<PoolGroup, 4> groups;
  for (auto &[window, pools] : byWindow) {
    
    // Sub-group by LHS shape to cleanly separate Layer 1 and Layer 2 pools
    // since they share the identical 2x2 window tensor.
    llvm::MapVector<Type, SmallVector<linalg::PoolingNchwSumOp>> byShape;
    for (auto pool : pools) {
      byShape[pool.getInputs()[0].getType()].push_back(pool);
    }

    for (auto &[shape, subPools] : byShape) {
      if (subPools.size() < 2) continue;

      SmallVector<Value> lhsVals;
      SmallVector<Value> outsVals;
      bool sameShape = true;
      
      auto firstLhsTy = cast<RankedTensorType>(subPools[0].getInputs()[0].getType());
      auto firstOutsTy = cast<RankedTensorType>(subPools[0].getOutputs()[0].getType());

      if (firstLhsTy.getDimSize(0) != 1) continue;

      for (auto pool : subPools) {
        Value lhs = pool.getInputs()[0];
        Value outs = pool.getOutputs()[0];
        lhsVals.push_back(lhs);
        outsVals.push_back(outs);

        if (cast<RankedTensorType>(lhs.getType()) != firstLhsTy ||
            cast<RankedTensorType>(outs.getType()) != firstOutsTy) {
          sameShape = false;
          break;
        }
      }
      if (!sameShape) continue;

      PoolGroup grp;
      grp.sharedWindow = window;
      grp.pools = subPools;
      grp.lhsVals = lhsVals;
      grp.outsVals = outsVals;
      groups.push_back(grp);
    }
  }
  return groups;
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct FTPPackingPass : public PassWrapper<FTPPackingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FTPPackingPass)

  StringRef getArgument() const final { return "ftp-packing"; }
  StringRef getDescription() const final { return "Batches SNN matmuls, convolutions, and poolings over unrolled timesteps."; }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<MatmulGroup, 4> matmulGroups = collectMatmulGroups(funcOp);
    SmallVector<ConvGroup, 4> convGroups = collectConvGroups(funcOp);
    SmallVector<PoolGroup, 4> poolGroups = collectPoolGroups(funcOp);

    if (matmulGroups.empty() && convGroups.empty() && poolGroups.empty()) return;
   
    OpBuilder builder(funcOp.getContext());

    // ---------------------------------------------------------
    // 1. Pack Convolutions
    // ---------------------------------------------------------
    for (ConvGroup &grp : convGroups) {
      Operation *insertionPt = grp.convs.back().getOperation();
      builder.setInsertionPointAfter(insertionPt);
      Location loc = insertionPt->getLoc();

      int64_t T = grp.T();
      
      auto lhsTy = grp.getLhsType();
      int64_t C = lhsTy.getDimSize(1);
      int64_t H = lhsTy.getDimSize(2);
      int64_t W = lhsTy.getDimSize(3);

      auto outsTy = grp.getOutsType();
      int64_t F = outsTy.getDimSize(1);
      int64_t oH = outsTy.getDimSize(2);
      int64_t oW = outsTy.getDimSize(3);

      // Stack LHS Inputs into [T, C, H, W]
      Value stackedLhs = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{T, C, H, W}, builder.getF32Type());
      for (int64_t t = 0; t < T; ++t) {
        SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(t), builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes   = {builder.getIndexAttr(1), builder.getIndexAttr(C), builder.getIndexAttr(H), builder.getIndexAttr(W)};
        SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1)};
        stackedLhs = builder.create<tensor::InsertSliceOp>(loc, grp.lhsVals[t], stackedLhs, offsets, sizes, strides);
      }

      // Stack Outs (Biases) into [T, F, oH, oW]
      Value stackedOuts = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{T, F, oH, oW}, builder.getF32Type());
      for (int64_t t = 0; t < T; ++t) {
        SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(t), builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes   = {builder.getIndexAttr(1), builder.getIndexAttr(F), builder.getIndexAttr(oH), builder.getIndexAttr(oW)};
        SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1)};
        stackedOuts = builder.create<tensor::InsertSliceOp>(loc, grp.outsVals[t], stackedOuts, offsets, sizes, strides);
      }

      // Execute Batched Conv
      auto batchedConvTy = RankedTensorType::get({T, F, oH, oW}, builder.getF32Type());
      auto batchedConv = builder.create<linalg::Conv2DNchwFchwOp>(
        loc, TypeRange{batchedConvTy}, ValueRange{stackedLhs, grp.sharedFilter}, ValueRange{stackedOuts}
      );
      batchedConv->setAttrs(grp.convs[0]->getAttrs()); // Copy strides/dilations

      // Extract slices and replace unrolled ops
      for (int64_t t = 0; t < T; ++t) {
        SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(t), builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes   = {builder.getIndexAttr(1), builder.getIndexAttr(F), builder.getIndexAttr(oH), builder.getIndexAttr(oW)};
        SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1)};
        Value slice = builder.create<tensor::ExtractSliceOp>(loc, batchedConv.getResult(0), offsets, sizes, strides);

        grp.convs[t].getResult(0).replaceAllUsesWith(slice);
        grp.convs[t].erase();
      }
    }

    // ---------------------------------------------------------
    // 2. Pack Pooling Layers
    // ---------------------------------------------------------
    for (PoolGroup &grp : poolGroups) {
      Operation *insertionPt = grp.pools.back().getOperation();
      builder.setInsertionPointAfter(insertionPt);
      Location loc = insertionPt->getLoc();

      int64_t T = grp.T();
      
      auto lhsTy = grp.getLhsType();
      int64_t C = lhsTy.getDimSize(1);
      int64_t H = lhsTy.getDimSize(2);
      int64_t W = lhsTy.getDimSize(3);

      auto outsTy = grp.getOutsType();
      int64_t oH = outsTy.getDimSize(2);
      int64_t oW = outsTy.getDimSize(3);

      // Stack LHS Inputs into [T, C, H, W]
      Value stackedLhs = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{T, C, H, W}, builder.getF32Type());
      for (int64_t t = 0; t < T; ++t) {
        SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(t), builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes   = {builder.getIndexAttr(1), builder.getIndexAttr(C), builder.getIndexAttr(H), builder.getIndexAttr(W)};
        SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1)};
        stackedLhs = builder.create<tensor::InsertSliceOp>(loc, grp.lhsVals[t], stackedLhs, offsets, sizes, strides);
      }

      // Stack Outs into [T, C, oH, oW]
      Value stackedOuts = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{T, C, oH, oW}, builder.getF32Type());
      for (int64_t t = 0; t < T; ++t) {
        SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(t), builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes   = {builder.getIndexAttr(1), builder.getIndexAttr(C), builder.getIndexAttr(oH), builder.getIndexAttr(oW)};
        SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1)};
        stackedOuts = builder.create<tensor::InsertSliceOp>(loc, grp.outsVals[t], stackedOuts, offsets, sizes, strides);
      }

      // Execute Batched Pooling
      auto batchedPoolTy = RankedTensorType::get({T, C, oH, oW}, builder.getF32Type());
      auto batchedPool = builder.create<linalg::PoolingNchwSumOp>(
        loc, TypeRange{batchedPoolTy}, ValueRange{stackedLhs, grp.sharedWindow}, ValueRange{stackedOuts}
      );
      batchedPool->setAttrs(grp.pools[0]->getAttrs()); // Copy strides/dilations

      // Extract slices and replace unrolled ops
      for (int64_t t = 0; t < T; ++t) {
        SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(t), builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes   = {builder.getIndexAttr(1), builder.getIndexAttr(C), builder.getIndexAttr(oH), builder.getIndexAttr(oW)};
        SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1)};
        Value slice = builder.create<tensor::ExtractSliceOp>(loc, batchedPool.getResult(0), offsets, sizes, strides);

        grp.pools[t].getResult(0).replaceAllUsesWith(slice);
        grp.pools[t].erase();
      }
    }

    // ---------------------------------------------------------
    // 3. Pack Matmuls
    // ---------------------------------------------------------
    for (MatmulGroup &grp : matmulGroups) {
      Operation *insertionPt = grp.matmuls.back().getOperation();
      builder.setInsertionPointAfter(insertionPt);
      Location loc = insertionPt->getLoc();

      int64_t T = grp.T();
      int64_t K = grp.K();
      int64_t N = grp.N();

      Value stacked = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{T, K}, builder.getF32Type());
      for (int64_t t = 0; t < T; ++t) {
        SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(t), builder.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes   = {builder.getIndexAttr(1), builder.getIndexAttr(K)};
        SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};
        stacked = builder.create<tensor::InsertSliceOp>(loc, grp.lhsVals[t], stacked, offsets, sizes, strides);
      }

      auto ftpOutType = RankedTensorType::get({T, N}, builder.getF32Type());
      Value zero = builder.create<arith::ConstantOp>(loc, builder.getF32Type(), builder.getF32FloatAttr(0.0f));
      Value ftpEmpty = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{T, N}, builder.getF32Type());
      Value ftpZero = builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{ftpEmpty}).getResult(0);

      Value ftpMatmul = builder.create<linalg::MatmulOp>(loc, TypeRange{ftpOutType}, ValueRange{stacked, grp.sharedRHS}, ValueRange{ftpZero}).getResult(0);

      Value finalBatched = ftpMatmul;
      if (grp.hasBias) {
        AffineMap identMap = AffineMap::getMultiDimIdentityMap(2, builder.getContext());
        AffineMap biasMap = AffineMap::get(2, 0, SmallVector<AffineExpr>{builder.getAffineDimExpr(1)}, builder.getContext());
        Value biasOut = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{T, N}, builder.getF32Type());

        finalBatched = builder.create<linalg::GenericOp>(
          loc, TypeRange{ftpOutType}, ValueRange{ftpMatmul, grp.bias}, ValueRange{biasOut},
          ArrayRef<AffineMap>{identMap, biasMap, identMap},
          ArrayRef<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::parallel},
          [&](OpBuilder &b, Location l, ValueRange args) {
            b.create<linalg::YieldOp>(l, b.create<arith::AddFOp>(l, args[0], args[1]).getResult());
          }).getResult(0);
      }

      for (int64_t t = 0; t < T; ++t) {
        SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(t), builder.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes   = {builder.getIndexAttr(1), builder.getIndexAttr(N)};
        SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};
        Value slice = builder.create<tensor::ExtractSliceOp>(loc, finalBatched, offsets, sizes, strides);

        if (grp.hasBias) {
          grp.biasOps[t].getResult(0).replaceAllUsesWith(slice);
          grp.biasOps[t].erase();
        } else {
          grp.matmuls[t].getResult(0).replaceAllUsesWith(slice);
        }
        grp.matmuls[t].erase();
      }
    }
   
    // 4. Clean up SSA Dominance
    if (!matmulGroups.empty() || !convGroups.empty() || !poolGroups.empty()) {
       sortBlockTopologically(&funcOp.getBody().front());
    }
  }
};
} // end namespace

namespace mlir {
namespace linalg {
std::unique_ptr<Pass> createFTPPackingPass() {
  return std::make_unique<FTPPackingPass>();
}
}
}