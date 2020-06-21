#pragma once

#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"
#include "onnx/onnxifi.h"

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/Eigen/Core"

#include <Eigen/Core>
#include <complex>
#include <unordered_map>
#include <vector>

namespace onnx_xla {
// Get access to XLA types used
using ::tensorflow::string;

using ::tensorflow::int16;
using ::tensorflow::int32;
using ::tensorflow::int64;
using ::tensorflow::int8;

using ::tensorflow::bfloat16;

using ::tensorflow::uint16;
using ::tensorflow::uint32;
using ::tensorflow::uint64;
using ::tensorflow::uint8;

using complex64 = std::complex<float>;

using ::Eigen::half;

using ::xla::PrimitiveType;
using ::xla::ShapeUtil;
using ::xla::XlaBuilder;
using ::xla::XlaComputation;
using ::xla::XlaOp;

using ::ONNX_NAMESPACE::Node;

// Helper functions to translate between ONNX and XLA types
PrimitiveType onnxToPrimitive(
    const ONNX_NAMESPACE::TensorProto_DataType& data_type);

// Utilities to help translation functions
XlaComputation add(PrimitiveType dataType);
XlaComputation max(PrimitiveType dataType);

std::vector<int64_t> parseOnnxInputSizes(const Node& n, size_t inputIndex);

std::vector<int64> getMultidirectionalBroadcastArg(const XlaBuilder& builder,
                                                   const XlaOp& firstOp,
                                                   const XlaOp& secondOp);
}  // namespace onnx_xla
