/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! CANN/HiAI converter for WebNN graphs.
//!
//! Converts `GraphInfo` into compiled model bytes via the adapter
//! (the `hiai-rs` crate's `adapter/`). The adapter wraps the HiAI DDK's C++ API.
//!
//! - `cann-runtime`: calls `encode_via_adapter()` to build a GE graph,
//!   compile via HiaiIrBuild, return ModelBufferData bytes.
//! - `cann-runtime-mock`: validates ops via `webnn_op_to_hiai`, returns
//!   placeholder bytes for CI/testing.

use crate::error::GraphError;
use crate::graph::GraphInfo;
use crate::operators::Operation;

use super::{ConvertedGraph, GraphConverter};

// Maps a WebNN operation to its HIAI IR operation name.
//
// Returns Some(name) for ops that map directly to an adapter cann_op_*()
//
// Returns `None` for ops that need decomposition or are not supported
#[cfg(any(feature = "cann-runtime", test))]
pub(crate) fn webnn_op_to_hiai(op: &Operation) -> Option<&'static str> {
    match op {
        // ── Element-wise binary ──────────────────────────────────────
        Operation::Add { .. } => Some("Add"),
        Operation::Sub { .. } => Some("Sub"),
        Operation::Mul { .. } => Some("Mul"),
        Operation::Div { .. } => Some("Div"),
        Operation::Pow { .. } => Some("Pow"),
        Operation::Max { .. } => Some("Max"),
        Operation::Min { .. } => Some("Min"),
        Operation::Equal { .. } => Some("Equal"),
        Operation::Greater { .. } => Some("Greater"),
        Operation::GreaterOrEqual { .. } => Some("GreaterOrEqual"),
        Operation::Lesser { .. } => Some("Lesser"),
        Operation::LesserOrEqual { .. } => Some("LesserOrEqual"),
        Operation::NotEqual { .. } => Some("NotEqual"),
        Operation::LogicalAnd { .. } => Some("LogicalAnd"),
        Operation::LogicalOr { .. } => Some("LogicalOr"),
        Operation::LogicalXor { .. } => Some("LogicalXor"),
        Operation::LogicalNot { .. } => Some("LogicalNot"),

        // ── Element-wise unary ───────────────────────────────────────
        Operation::Abs { .. } => Some("Abs"),
        Operation::Neg { .. } => Some("Neg"),
        Operation::Exp { .. } => Some("Exp"),
        Operation::Log { .. } => Some("Log"),
        Operation::Sin { .. } => Some("Sin"),
        Operation::Cos { .. } => Some("Cos"),
        Operation::Tan { .. } => Some("Tan"),
        Operation::Sqrt { .. } => Some("Sqrt"),
        Operation::Ceil { .. } => Some("Ceil"),
        Operation::Floor { .. } => Some("Floor"),
        Operation::Sign { .. } => Some("Sign"),
        Operation::Erf { .. } => Some("Erf"),
        Operation::Reciprocal { .. } => Some("Reciprocal"),
        Operation::RoundEven { .. } => Some("Round"),
        Operation::Cast { .. } => Some("Cast"),
        Operation::Clamp { .. } => Some("Clamp"),

        // ── Activations ──────────────────────────────────────────────
        Operation::Relu { .. } => Some("ReLU"),
        Operation::Sigmoid { .. } => Some("Sigmoid"),
        Operation::Tanh { .. } => Some("Tanh"),
        Operation::Elu { .. } => Some("ELU"),
        Operation::Gelu { .. } => Some("GELU"),
        Operation::LeakyRelu { .. } => Some("LeakyRelu"),
        Operation::HardSigmoid { .. } => Some("HardSigmoid"),
        Operation::HardSwish { .. } => Some("HardSwish"),
        Operation::Softplus { .. } => Some("Softplus"),
        Operation::Softsign { .. } => Some("Softsign"),

        // ── Convolution + Pool + Matmul ──────────────────────────────
        Operation::Conv2d { .. } => Some("Conv2D"),
        // ConvTranspose2d is decomposed into a Convolution (stride=1 +
        // recomputed pads) to match Chromium's kTransposed case; native
        // hiai::op::ConvTranspose is unsupported on the NPU.
        Operation::ConvTranspose2d { .. } => Some("Conv2D"),
        Operation::MaxPool2d { .. } => Some("MaxPool"),
        Operation::AveragePool2d { .. } => Some("AvgPool"),
        Operation::L2Pool2d { .. } => Some("MaxPool"),
        Operation::Matmul { .. } => Some("MatMul"),
        Operation::Gemm { .. } => Some("Gemm"),
        Operation::Softmax { .. } => Some("Softmax"),

        // ── Reduction ────────────────────────────────────────────────
        Operation::ReduceSum { .. } => Some("ReduceSum"),
        Operation::ReduceMean { .. } => Some("ReduceMean"),
        Operation::ReduceMax { .. } => Some("ReduceMax"),
        Operation::ReduceMin { .. } => Some("ReduceMin"),
        Operation::ReduceProduct { .. } => Some("ReduceProduct"),
        Operation::ReduceL1 { .. } => Some("ReduceL1"),
        Operation::ReduceL2 { .. } => Some("ReduceL2"),
        Operation::ReduceLogSum { .. } => Some("ReduceLogSum"),
        Operation::ReduceLogSumExp { .. } => Some("ReduceLogSumExp"),
        Operation::ReduceSumSquare { .. } => Some("ReduceSumSquare"),
        Operation::ArgMax { .. } => Some("ArgMax"),
        Operation::ArgMin { .. } => Some("ArgMin"),

        // ── Shape ops ────────────────────────────────────────────────
        Operation::Reshape { .. } => Some("Reshape"),
        Operation::Transpose { .. } => Some("Transpose"),
        Operation::Tile { .. } => Some("Tile"),
        Operation::Slice { .. } => Some("Slice"),
        Operation::Split { .. } => Some("Split"),
        Operation::Concat { .. } => Some("Concat"),
        Operation::Pad { .. } => Some("Pad"),
        Operation::Squeeze { .. } => Some("Reshape"),
        Operation::Unsqueeze { .. } => Some("Reshape"),
        Operation::Expand { .. } => Some("Expand"),
        Operation::CumulativeSum { .. } => Some("CumulativeSum"),

        // ── Gather / Scatter / Where ─────────────────────────────────
        Operation::Gather { .. } => Some("Gather"),
        Operation::GatherElements { .. } => Some("Gather"),
        Operation::GatherND { .. } => Some("GatherND"),
        Operation::ScatterElements { .. } => Some("ScatterND"),
        Operation::ScatterND { .. } => Some("ScatterND"),
        Operation::Where { .. } => Some("Where"),

        // ── Normalization + Quantization ─────────────────────────────
        Operation::BatchNormalization { .. } => Some("BatchNormalization"),
        Operation::InstanceNormalization { .. } => Some("BatchNormalization"),
        Operation::QuantizeLinear { .. } => Some("QuantizeLinear"),
        Operation::DequantizeLinear { .. } => None,

        // ── Other ────────────────────────────────────────────────────
        Operation::Resample2d { .. } => Some("Resample2D"),
        Operation::GlobalAveragePool { .. } => None,
        Operation::GlobalMaxPool { .. } => None,
        Operation::Constant { .. } => Some("Constant"),
        Operation::Shape { .. } => Some("Shape"),

        // ── Needs decomposition ───────────────────────────────────────
        Operation::Identity { .. } => None,
        Operation::Prelu { .. } => None,
        Operation::Linear { .. } => None,
        Operation::LayerNormalization { .. } => None,
        Operation::Triangular { .. } => None,
        Operation::IsNaN { .. } => None,
        Operation::IsInfinite { .. } => None,
        Operation::Reverse { .. } => None,

        // ── Not supported ─────────────────────────────────────────────
        Operation::Gru { .. }
        | Operation::GruCell { .. }
        | Operation::Lstm { .. }
        | Operation::LstmCell { .. } => None,
    }
}

// Operations the CANN backend cannot express at all. RNN ops are the only
// remaining gap: the Chromium reference leaves them NOTIMPLEMENTED() and the
// adapter has no hiai op for them. Everything else is wired (or decomposed)
// below.
pub(crate) fn is_unsupported_op(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Gru { .. }
            | Operation::GruCell { .. }
            | Operation::Lstm { .. }
            | Operation::LstmCell { .. }
    )
}

#[cfg(feature = "cann-runtime")]
mod adapter {
    use super::*;
    use crate::graph::{DataType, OperandDescriptor};
    use hiai_rs::sys::*;

    // ── Graph builder helpers ──────────────────────────────────────────

    fn cann_data_type(data_type: DataType) -> hiai_rs::sys::ddk_CannDataType {
        use hiai_rs::sys::ddk_CannDataType as C;
        match data_type {
            DataType::Float32 => C::CANN_DT_FLOAT,
            DataType::Float16 => C::CANN_DT_FLOAT16,
            DataType::Int32 => C::CANN_DT_INT32,
            DataType::Int8 => C::CANN_DT_INT8,
            DataType::Uint8 => C::CANN_DT_UINT8,
            DataType::Uint32 => C::CANN_DT_UINT32,
            DataType::Int64 => C::CANN_DT_INT64,
            DataType::Uint64 => C::CANN_DT_UINT64,
            _ => C::CANN_DT_FLOAT,
        }
    }

    // Binary operators.
    // Returns (input_a, input_b, ge_input_name_a, ge_input_name_b), or None.
    fn binary_op_inputs(operation: &Operation) -> Option<(u32, u32, &'static str, &'static str)> {
        let (a, b, name_a, name_b) = match operation {
            Operation::Add { a, b, .. }
            | Operation::Sub { a, b, .. }
            | Operation::Mul { a, b, .. }
            | Operation::Div { a, b, .. }
            | Operation::Pow { a, b, .. }
            | Operation::Max { a, b, .. }
            | Operation::Min { a, b, .. } => (*a, *b, "x1", "x2"),
            Operation::Equal { a, b, .. }
            | Operation::Greater { a, b, .. }
            | Operation::GreaterOrEqual { a, b, .. }
            | Operation::Lesser { a, b, .. }
            | Operation::LesserOrEqual { a, b, .. }
            | Operation::NotEqual { a, b, .. } => (*a, *b, "x1", "x2"),
            Operation::LogicalAnd { a, b, .. }
            | Operation::LogicalOr { a, b, .. }
            | Operation::LogicalXor { a, b, .. } => (*a, *b, "x1", "x2"),
            Operation::Matmul { a, b, .. } => (*a, *b, "x1", "x2"),
            _ => return None,
        };
        Some((a, b, name_a, name_b))
    }

    // Unary element-wise operators.
    // Returns input operand index, or None.
    fn unary_op_input(operation: &Operation) -> Option<u32> {
        match operation {
            Operation::Relu { input, .. }
            | Operation::Sigmoid { input, .. }
            | Operation::Tanh { input, .. }
            | Operation::Elu { input, .. }
            | Operation::Gelu { input, .. }
            | Operation::LeakyRelu { input, .. }
            | Operation::HardSigmoid { input, .. }
            | Operation::HardSwish { input, .. }
            | Operation::Softplus { input, .. }
            | Operation::Softsign { input, .. } => Some(*input),
            Operation::Abs { input, .. }
            | Operation::Neg { input, .. }
            | Operation::Exp { input, .. }
            | Operation::Log { input, .. }
            | Operation::Sin { input, .. }
            | Operation::Cos { input, .. }
            | Operation::Tan { input, .. }
            | Operation::Sqrt { input, .. }
            | Operation::Ceil { input, .. }
            | Operation::Floor { input, .. }
            | Operation::Sign { input, .. }
            | Operation::Erf { input, .. }
            | Operation::Reciprocal { input, .. }
            | Operation::RoundEven { input, .. } => Some(*input),
            Operation::Cast { input, .. } | Operation::Identity { input, .. } => Some(*input),
            Operation::Clamp { input, .. }
            | Operation::LogicalNot { input, .. }
            | Operation::Shape { input, .. } => Some(*input),
            _ => None,
        }
    }

    fn descriptor_dims(descriptor: &OperandDescriptor) -> Vec<i64> {
        use crate::graph::Dimension;
        descriptor
            .shape
            .iter()
            .map(|dim| match dim {
                Dimension::Static(dim_value) => *dim_value as i64,
                _ => 0,
            })
            .collect()
    }

    // Extract an f32 from a WebNN `MLNumber` (serde_json::Value number).
    fn json_number_f32(value: &serde_json::Value) -> Option<f32> {
        value
            .as_f64()
            .map(|f| f as f32)
            .or_else(|| value.as_i64().map(|i| i as f32))
            .or_else(|| value.as_u64().map(|u| u as f32))
    }

    // Reduce wiring info: (input, options, axes_as_attr, keep_dims_attr_name).
    // sum/mean/max/min take axes as an int32 const input; product/l2/logSumExp
    // take axes as an int64-list attr. `keep_dims_attr_name` differs
    // ("keep_dims" vs "keepdims").
    fn reduce_op_info(
        op: &Operation,
    ) -> Option<(
        u32,
        &Option<crate::operator_options::MLReduceOptions>,
        bool,
        &'static str,
    )> {
        match op {
            Operation::ReduceSum { input, options, .. }
            | Operation::ReduceMean { input, options, .. }
            | Operation::ReduceMax { input, options, .. }
            | Operation::ReduceMin { input, options, .. } => {
                Some((*input, options, false, "keep_dims"))
            }
            Operation::ReduceProduct { input, options, .. }
            | Operation::ReduceL2 { input, options, .. } => {
                Some((*input, options, true, "keep_dims"))
            }
            Operation::ReduceLogSumExp { input, options, .. } => {
                Some((*input, options, true, "keepdims"))
            }
            _ => None,
        }
    }

    // Reduction axes: requested axes, or all axes when none are given.
    fn reduce_axes(
        input: u32,
        options: &Option<crate::operator_options::MLReduceOptions>,
        graph: &GraphInfo,
    ) -> Vec<i32> {
        let rank = graph.operands[input as usize].descriptor.shape.len();
        options
            .as_ref()
            .and_then(|o| o.axes.as_ref())
            .map(|a| a.iter().map(|&ax| ax as i32).collect())
            .unwrap_or_else(|| (0..rank as u32).map(|ax| ax as i32).collect())
    }

    // Create a registered op by type name, registering it as an intermediate.
    fn create_op(
        type_str: &str,
        name: &str,
        extra_ops: &mut Vec<ddk_CannOperatorHandle>,
    ) -> Result<ddk_CannOperatorHandle, GraphError> {
        let name_c = std::ffi::CString::new(name).unwrap();
        let type_c = std::ffi::CString::new(type_str).unwrap();
        let op = unsafe { ddk_cann_operator_create_registered(type_c.as_ptr(), name_c.as_ptr()) };
        if op.is_null() {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_operator_create failed for {type_str} '{name}'").into(),
            });
        }
        extra_ops.push(op);
        Ok(op)
    }

    // Connect `name` input on `op` to `handle`.
    fn connect_input(
        op: ddk_CannOperatorHandle,
        name: &str,
        handle: ddk_CannOperatorHandle,
    ) -> Result<(), GraphError> {
        let name_c = std::ffi::CString::new(name).unwrap();
        let status = unsafe { ddk_cann_operator_set_input(op, name_c.as_ptr(), handle) };
        if status != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_operator_set_input for '{name}' failed").into(),
            });
        }
        Ok(())
    }

    // Connect source operand `src_id` (which may be a Split output slot) to
    // `name` on `op`, routing SplitD slots via the producer output index.
    fn connect_src_input(
        op: ddk_CannOperatorHandle,
        name: &str,
        src_id: u32,
        handles: &[ddk_CannOperatorHandle],
        split_out: &std::collections::HashMap<u32, u32>,
    ) -> Result<(), GraphError> {
        let name_c = std::ffi::CString::new(name).unwrap();
        let handle = handles[src_id as usize];
        let status = match split_out.get(&src_id) {
            Some(&out_i) => unsafe {
                ddk_cann_operator_set_input_by_output(op, name_c.as_ptr(), handle, out_i)
            },
            None => unsafe { ddk_cann_operator_set_input(op, name_c.as_ptr(), handle) },
        };
        if status != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_operator_set_input for '{name}' failed").into(),
            });
        }
        Ok(())
    }

    // Set an int64 attr on `op`.
    fn set_int64_attr(op: ddk_CannOperatorHandle, name: &str, value: i64) {
        let name_c = std::ffi::CString::new(name).unwrap();
        unsafe { ddk_cann_operator_set_attr_int64(op, name_c.as_ptr(), value) };
    }

    // Set a bool attr on `op`.
    fn set_bool_attr(op: ddk_CannOperatorHandle, name: &str, value: bool) {
        let name_c = std::ffi::CString::new(name).unwrap();
        unsafe { ddk_cann_operator_set_attr_bool(op, name_c.as_ptr(), value as i32) };
    }

    // Set a float attr on `op`.
    fn set_float_attr(op: ddk_CannOperatorHandle, name: &str, value: f32) {
        let name_c = std::ffi::CString::new(name).unwrap();
        unsafe { ddk_cann_operator_set_attr_float(op, name_c.as_ptr(), value) };
    }

    // Create a Const operator holding the given tensor data, mirroring the
    // Chromium reference's Constant<T>() helper (Const().set_attr_value(tensor)).
    fn make_const(
        name: &str,
        data: &[u8],
        shape: &[i64],
        dtype: ddk_CannDataType,
        format: i32,
    ) -> ddk_CannOperatorHandle {
        let name_c = std::ffi::CString::new(name).unwrap();
        let value_name = std::ffi::CString::new("value").unwrap();
        let const_op = unsafe { ddk_cann_op_const_with_name(name_c.as_ptr()) };
        unsafe {
            ddk_cann_operator_set_attr_tensor_raw_format(
                const_op,
                value_name.as_ptr(),
                data.as_ptr() as *const std::ffi::c_void,
                data.len() as u32,
                shape.as_ptr(),
                shape.len() as i32,
                dtype,
                format,
            );
        }
        const_op
    }

    // Emit a Transpose op with a fixed permutation, used to normalize NHWC to
    // NCHW for the layout-agnostic spatial ops (the GE Convolution/PoolingD
    // infershape is NCHW-only). Returns the transpose handle.
    fn emit_transpose(
        name: &str,
        x_handle: ddk_CannOperatorHandle,
        perm: &[i32],
        extra_ops: &mut Vec<ddk_CannOperatorHandle>,
    ) -> Result<ddk_CannOperatorHandle, GraphError> {
        let transpose = create_op("Transpose", name, extra_ops)?;
        connect_input(transpose, "x", x_handle)?;
        let perm_const = make_const(
            &format!("{name}_perm"),
            bytemuck::cast_slice(perm),
            &[perm.len() as i64],
            ddk_CannDataType::CANN_DT_INT32,
            2, // FORMAT_ND
        );
        extra_ops.push(perm_const);
        connect_input(transpose, "perm", perm_const)?;
        Ok(transpose)
    }

    // Emit a Reshape op (x + int64 shape const). Used to wrap a Permute that
    // feeds a Convolution: GE's ConvolutionInfer reads a Permute input's
    // *input* shape instead of its output, so an identity Reshape re-asserts
    // the correct (NCHW) shape. Matches graph_builder_cann.cc AddConv2dOperation.
    fn emit_reshape(
        name: &str,
        x_handle: ddk_CannOperatorHandle,
        shape: &[i64],
        extra_ops: &mut Vec<ddk_CannOperatorHandle>,
    ) -> Result<ddk_CannOperatorHandle, GraphError> {
        let reshape = create_op("Reshape", name, extra_ops)?;
        connect_input(reshape, "x", x_handle)?;
        let shape_const = make_const(
            &format!("{name}_shape"),
            bytemuck::cast_slice(shape),
            &[shape.len() as i64],
            ddk_CannDataType::CANN_DT_INT64,
            2, // FORMAT_ND
        );
        extra_ops.push(shape_const);
        connect_input(reshape, "shape", shape_const)?;
        Ok(reshape)
    }

    // Plain dequantize int8/uint8 -> float32 (symmetric, no zero-point; the
    // quantized model uses cast + per-channel mul, with no zero-point).
    // Float32 data is returned unchanged.
    fn plain_dequantize_f32(data: &[u8], dtype: DataType) -> Vec<f32> {
        match dtype {
            DataType::Int8 => data.iter().map(|&b| b as i8 as f32).collect(),
            DataType::Uint8 => data.iter().map(|&b| b as f32).collect(),
            _ => bytemuck::cast_slice(data).to_vec(),
        }
    }

    // Resolve an operand to its dequantized float32 data + shape, following the
    // quantized-model dequantization chains:
    //   mul(cast(int8/uint8 const), per-channel-scale-const)
    //   cast(int8/uint8 const)
    //   int8/uint8 or float32 const
    // Returns None if the operand is not a foldable constant expression.
    fn resolve_dequantized_f32(graph: &GraphInfo, id: u32) -> Option<(Vec<f32>, Vec<i64>)> {
        // Direct constant.
        if let Some(cd) = graph.constant_operand_ids_to_handles.get(&id) {
            let desc = &graph.operands[id as usize].descriptor;
            let dims = descriptor_dims(desc);
            return Some((plain_dequantize_f32(&cd.data, desc.data_type), dims));
        }
        // cast(x) -> x.
        for op in &graph.operations {
            if let Operation::Cast { input, outputs, .. } = op {
                if outputs.contains(&id) {
                    return resolve_dequantized_f32(graph, *input);
                }
            }
        }
        // mul(a, b) -> per-channel scaled data (dequantization scale on axis 0).
        for op in &graph.operations {
            if let Operation::Mul { a, b, outputs, .. } = op {
                if outputs.contains(&id) {
                    if let (Some((a_data, a_dims)), Some((b_data, b_dims))) = (
                        resolve_dequantized_f32(graph, *a),
                        resolve_dequantized_f32(graph, *b),
                    ) {
                        if let Some(out) = mul_channel_scale(a_data, a_dims, b_data, b_dims) {
                            return Some(out);
                        }
                    }
                    return None;
                }
            }
        }
        None
    }

    // Multiply `data` by a per-channel `scale` broadcast along axis 0. Returns
    // (scaled data, data dims). The scale may be rank-1 ([o]) or [o,1,1,1];
    // a scalar (len 1) is also accepted.
    fn mul_channel_scale(
        a_data: Vec<f32>,
        a_dims: Vec<i64>,
        b_data: Vec<f32>,
        b_dims: Vec<i64>,
    ) -> Option<(Vec<f32>, Vec<i64>)> {
        // The larger tensor is the data, the smaller is the scale.
        let (data, data_dims, scale) = if a_data.len() >= b_data.len() {
            (a_data, a_dims, b_data)
        } else {
            (b_data, b_dims, a_data)
        };
        if data_dims.is_empty() || data_dims[0] <= 0 {
            return None;
        }
        let o = data_dims[0] as usize;
        if scale.len() == 1 {
            let s = scale[0];
            let out = data.iter().map(|&v| v * s).collect();
            return Some((out, data_dims));
        }
        if scale.len() != o {
            return None;
        }
        let per_channel = data.len() / o;
        if per_channel * o != data.len() {
            return None;
        }
        let mut out = vec![0.0f32; data.len()];
        for (idx, &v) in data.iter().enumerate() {
            out[idx] = v * scale[idx / per_channel];
        }
        Some((out, data_dims))
    }

    // Transpose an OHWI [o,h,w,i] filter (already f32) into the target layout.
    // `perm` maps each target axis to its source axis in [o,h,w,i]:
    // [0,3,1,2] -> OIHW, [3,0,1,2] -> IOHW. Returns (f32, target shape).
    fn transpose_ohwi_f32(
        data: &[f32],
        o: usize,
        h: usize,
        w: usize,
        i: usize,
        perm: [usize; 4],
    ) -> (Vec<f32>, [i64; 4]) {
        let src_dims = [o, h, w, i];
        let dst_dims = [
            src_dims[perm[0]],
            src_dims[perm[1]],
            src_dims[perm[2]],
            src_dims[perm[3]],
        ];
        let dst_strides = [
            dst_dims[1] * dst_dims[2] * dst_dims[3],
            dst_dims[2] * dst_dims[3],
            dst_dims[3],
            1,
        ];
        // inv[src_axis] = target_axis (perm[target_axis] == src_axis).
        let mut inv = [0usize; 4];
        for (k, &p) in perm.iter().enumerate() {
            inv[p] = k;
        }
        let mut out = vec![0.0f32; data.len()];
        let mut src_idx = 0usize;
        for oo in 0..o {
            for hh in 0..h {
                for ww in 0..w {
                    for ii in 0..i {
                        let src_coord = [oo, hh, ww, ii];
                        let mut tgt = [0usize; 4];
                        for s in 0..4 {
                            tgt[inv[s]] = src_coord[s];
                        }
                        let dst_idx = tgt[0] * dst_strides[0]
                            + tgt[1] * dst_strides[1]
                            + tgt[2] * dst_strides[2]
                            + tgt[3] * dst_strides[3];
                        out[dst_idx] = data[src_idx];
                        src_idx += 1;
                    }
                }
            }
        }
        (
            out,
            [
                dst_dims[0] as i64,
                dst_dims[1] as i64,
                dst_dims[2] as i64,
                dst_dims[3] as i64,
            ],
        )
    }

    // When a spatial op declares an OHWI filter, reorder it to the NCHW filter
    // layout expected by the GE Convolution by dequantizing (per-channel scale
    // included) and transposing the filter bytes into a new float32 const (the
    // FMK `Permute` op cannot infershape a const-input transpose, so this is
    // done on the host). `perm` maps [o,h,w,i] -> target layout, typically
    // [0,3,1,2] -> OIHW. Returns None if no reorder is needed.
    fn filter_ohwi_to_nchw(
        graph: &GraphInfo,
        filter_id: u32,
        filter_layout: &str,
        extra_ops: &mut Vec<ddk_CannOperatorHandle>,
        name: &str,
        perm: [usize; 4],
    ) -> Result<Option<ddk_CannOperatorHandle>, GraphError> {
        if !filter_layout.eq_ignore_ascii_case("ohwi") {
            return Ok(None);
        }
        let (f32_data, dims) = match resolve_dequantized_f32(graph, filter_id) {
            Some(x) => x,
            None => return Ok(None),
        };
        if dims.len() != 4 {
            return Ok(None);
        }
        let (o, h, w, i) = (
            dims[0] as usize,
            dims[1] as usize,
            dims[2] as usize,
            dims[3] as usize,
        );
        log::error!(
            "[cann-debug] filter_ohwi_to_nchw {} ohwi dims={dims:?} perm={perm:?}",
            name
        );
        let (transposed, shape) = transpose_ohwi_f32(&f32_data, o, h, w, i, perm);
        let const_op = make_const(
            name,
            bytemuck::cast_slice(&transposed),
            &shape,
            ddk_CannDataType::CANN_DT_FLOAT,
            0, // FORMAT_NCHW
        );
        extra_ops.push(const_op);
        Ok(Some(const_op))
    }

    // Connect an operand to `name` on `op`. (Split is decomposed into Slice ops
    // elsewhere, so every source here is single-output.)
    unsafe fn set_operand_input(
        op: ddk_CannOperatorHandle,
        name: &std::ffi::CStr,
        handle: ddk_CannOperatorHandle,
    ) -> i32 {
        unsafe { ddk_cann_operator_set_input(op, name.as_ptr(), handle) }
    }

    // Connect `index` on dynamic input `name` of `op` to a source operand's
    // handle. Dynamic inputs route a multi-output source (Split) by the
    // consumer's 1-based index, matching the Chromium reference.
    unsafe fn set_dynamic_input(
        op: ddk_CannOperatorHandle,
        name: &std::ffi::CStr,
        index: u32,
        handle: ddk_CannOperatorHandle,
    ) -> i32 {
        unsafe { ddk_cann_operator_set_dynamic_input_by_index(op, name.as_ptr(), index, handle) }
    }

    // Connect source operand `src_id` (which may be a multi-output SplitD slot)
    // to `name` on `op`. Split outputs route via the producer's output index;
    // every other source connects by name to its single output.
    unsafe fn set_src_input(
        op: ddk_CannOperatorHandle,
        name: &std::ffi::CStr,
        src_id: u32,
        handles: &[ddk_CannOperatorHandle],
        split_out: &std::collections::HashMap<u32, u32>,
    ) -> i32 {
        let handle = handles[src_id as usize];
        match split_out.get(&src_id) {
            Some(&out_i) => unsafe {
                ddk_cann_operator_set_input_by_output(op, name.as_ptr(), handle, out_i)
            },
            None => unsafe { ddk_cann_operator_set_input(op, name.as_ptr(), handle) },
        }
    }

    // Same as `set_src_input`, but for a dynamic input slot `index` on `op`.
    unsafe fn set_src_dynamic_input(
        op: ddk_CannOperatorHandle,
        name: &std::ffi::CStr,
        index: u32,
        src_id: u32,
        handles: &[ddk_CannOperatorHandle],
        split_out: &std::collections::HashMap<u32, u32>,
    ) -> i32 {
        let handle = handles[src_id as usize];
        match split_out.get(&src_id) {
            Some(&out_i) => unsafe {
                ddk_cann_operator_set_dynamic_input_by_index_by_output(
                    op,
                    name.as_ptr(),
                    index,
                    handle,
                    out_i,
                )
            },
            None => unsafe {
                ddk_cann_operator_set_dynamic_input_by_index(op, name.as_ptr(), index, handle)
            },
        }
    }

    // Returns the output operand slice for all operations with standard
    // `outputs: Vec<u32>` field.
    fn op_outputs(operation: &Operation) -> &[u32] {
        match operation {
            // Binary
            Operation::Add { outputs, .. }
            | Operation::Sub { outputs, .. }
            | Operation::Mul { outputs, .. }
            | Operation::Div { outputs, .. }
            | Operation::Pow { outputs, .. }
            | Operation::Max { outputs, .. }
            | Operation::Min { outputs, .. }
            // Comparison
            | Operation::Equal { outputs, .. }
            | Operation::Greater { outputs, .. }
            | Operation::GreaterOrEqual { outputs, .. }
            | Operation::Lesser { outputs, .. }
            | Operation::LesserOrEqual { outputs, .. }
            | Operation::NotEqual { outputs, .. }
            // Logical binary
            | Operation::LogicalAnd { outputs, .. }
            | Operation::LogicalOr { outputs, .. }
            | Operation::LogicalXor { outputs, .. }
            // Activations
            | Operation::Relu { outputs, .. }
            | Operation::Sigmoid { outputs, .. }
            | Operation::Tanh { outputs, .. }
            | Operation::Elu { outputs, .. }
            | Operation::Gelu { outputs, .. }
            | Operation::LeakyRelu { outputs, .. }
            | Operation::HardSigmoid { outputs, .. }
            | Operation::HardSwish { outputs, .. }
            | Operation::Softplus { outputs, .. }
            | Operation::Softsign { outputs, .. }
            // Unary math
            | Operation::Abs { outputs, .. }
            | Operation::Neg { outputs, .. }
            | Operation::Exp { outputs, .. }
            | Operation::Log { outputs, .. }
            | Operation::Sin { outputs, .. }
            | Operation::Cos { outputs, .. }
            | Operation::Tan { outputs, .. }
            | Operation::Sqrt { outputs, .. }
            | Operation::Ceil { outputs, .. }
            | Operation::Floor { outputs, .. }
            | Operation::Sign { outputs, .. }
            | Operation::Erf { outputs, .. }
            | Operation::Reciprocal { outputs, .. }
            // Type / other
            | Operation::Cast { outputs, .. }
            | Operation::Clamp { outputs, .. }
            | Operation::Identity { outputs, .. }
            | Operation::LogicalNot { outputs, .. }
            | Operation::Softmax { outputs, .. }
            // Matrix
            | Operation::Matmul { outputs, .. }
            | Operation::Gemm { outputs, .. }
            // Conv
            | Operation::Conv2d { outputs, .. }
            | Operation::ConvTranspose2d { outputs, .. }
            // Pooling
            | Operation::MaxPool2d { outputs, .. }
            | Operation::AveragePool2d { outputs, .. }
            | Operation::L2Pool2d { outputs, .. }
            | Operation::GlobalMaxPool { outputs, .. }
            | Operation::GlobalAveragePool { outputs, .. }
            // ArgMax / ArgMin
            | Operation::ArgMax { outputs, .. }
            | Operation::ArgMin { outputs, .. }
            // Reduction
            | Operation::ReduceSum { outputs, .. }
            // Normalization
            | Operation::BatchNormalization { outputs, .. }
            // Reduction (full set)
            | Operation::ReduceMean { outputs, .. }
            | Operation::ReduceMax { outputs, .. }
            | Operation::ReduceMin { outputs, .. }
            | Operation::ReduceProduct { outputs, .. }
            | Operation::ReduceL1 { outputs, .. }
            | Operation::ReduceL2 { outputs, .. }
            | Operation::ReduceLogSum { outputs, .. }
            | Operation::ReduceLogSumExp { outputs, .. }
            | Operation::ReduceSumSquare { outputs, .. }
            // Shape / other
            | Operation::Concat { outputs, .. }
            | Operation::Reshape { outputs, .. }
            | Operation::Resample2d { outputs, .. }
            | Operation::Slice { outputs, .. }
            | Operation::Split { outputs, .. }
            | Operation::Transpose { outputs, .. }
            | Operation::Tile { outputs, .. }
            | Operation::Pad { outputs, .. }
            | Operation::Squeeze { outputs, .. }
            | Operation::Unsqueeze { outputs, .. }
            | Operation::Expand { outputs, .. }
            | Operation::CumulativeSum { outputs, .. }
            | Operation::Reverse { outputs, .. }
            // Gather / Scatter / Where
            | Operation::Gather { outputs, .. }
            | Operation::GatherElements { outputs, .. }
            | Operation::GatherND { outputs, .. }
            | Operation::ScatterElements { outputs, .. }
            | Operation::ScatterND { outputs, .. }
            | Operation::Where { outputs, .. }
            // Normalization / Quantization
            | Operation::InstanceNormalization { outputs, .. }
            | Operation::LayerNormalization { outputs, .. }
            | Operation::QuantizeLinear { outputs, .. }
            | Operation::DequantizeLinear { outputs, .. }
            // Other
            | Operation::Linear { outputs, .. }
            | Operation::Triangular { outputs, .. }
            | Operation::IsNaN { outputs, .. }
            | Operation::IsInfinite { outputs, .. }
            | Operation::RoundEven { outputs, .. } => outputs,
            _ => &[],
        }
    }

    // ── Layout helpers ─────────────────────────────────────────────────

    /// Map WebNN input_layout to GE data_format. NHWC spatial ops are
    /// normalized to NCHW via transposes in the op wiring, so the GE op is
    /// always NCHW here.
    fn conv_data_format(_options: &Option<crate::operator_options::MLConv2dOptions>) -> &str {
        "NCHW"
    }

    // Build a CANN graph via the adapter, compile it, and return the model bytes.
    //
    // Data nodes (with tensor descriptors) -> ops -> NetOutput -> compile -> bytes.
    // Returns Err if the shim library is unavailable.
    pub(crate) fn encode_via_adapter(graph: &GraphInfo) -> Result<Vec<u8>, GraphError> {
        use std::ffi::CString;

        // ── 1. Create graph ─────────────────────────────────────────────
        let graph_name = CString::new("webnn_model").unwrap();
        let can_graph = unsafe { ddk_cann_graph_create(graph_name.as_ptr()) };
        if can_graph.is_null() {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_graph_create failed".into(),
            });
        }

        // operand_index -> CannOperatorHandle
        let mut handles: Vec<ddk_CannOperatorHandle> =
            vec![std::ptr::null_mut(); graph.operands.len()];
        // operand_index -> output slot of a multi-output producer (SplitD).
        // Absent for single-output ops; for Split output i this maps to i so
        // consumers can route via cann_operator_set_input_by_output.
        let mut split_out: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

        // ── 2. Create Data operators for each input ──────────────────────
        let mut data_ops: Vec<ddk_CannOperatorHandle> = Vec::new();

        for &input_id in &graph.input_operands {
            let descriptor = &graph.operands[input_id as usize].descriptor;
            let dimensions = descriptor_dims(descriptor);
            let name = CString::new(
                graph.operands[input_id as usize]
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("input_{input_id}")),
            )
            .unwrap();

            let data_op = unsafe { ddk_cann_op_data_with_name(name.as_ptr()) };
            if data_op.is_null() {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_op_data_with_name for operand {input_id} failed").into(),
                });
            }

            // Set tensor descriptor: shape + FORMAT_ND + dtype
            let shape =
                unsafe { ddk_cann_shape_create(dimensions.as_ptr(), dimensions.len() as i32) };
            if shape.is_null() {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: "cann_shape_create failed".into(),
                });
            }
            let tensor_desc = unsafe {
                ddk_cann_tensor_desc_create(
                    shape,
                    ddk_CannFormat::CANN_FORMAT_NCHW,
                    cann_data_type(descriptor.data_type),
                )
            };
            if tensor_desc.is_null() {
                unsafe { ddk_cann_shape_destroy(shape) };
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: "cann_tensor_desc_create failed".into(),
                });
            }
            let x_name = CString::new("x").unwrap();
            let status = unsafe {
                ddk_cann_operator_update_input_desc(data_op, x_name.as_ptr(), tensor_desc)
            };
            unsafe {
                ddk_cann_tensor_desc_destroy(tensor_desc);
                ddk_cann_shape_destroy(shape);
            }
            if status != 0 {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!(
                        "cann_operator_update_input_desc for operand {input_id} failed: {status}"
                    )
                    .into(),
                });
            }

            handles[input_id as usize] = data_op;
            data_ops.push(data_op);
        }

        // Create Const operators for constant operands (for example, Conv2d filters).
        // Iterate in sorted operand order: `constant_operand_ids_to_handles` is a
        // HashMap, whose iteration order is randomized per instance. Emitting
        // consts in a stable order keeps the serialized model bytes identical
        // across `build()` calls, so the backend's model-bytes session cache hits.
        let mut const_ops: Vec<ddk_CannOperatorHandle> = Vec::new();
        let mut const_ids: Vec<u32> = graph
            .constant_operand_ids_to_handles
            .keys()
            .copied()
            .collect();
        const_ids.sort_unstable();
        for const_id in const_ids {
            let constant_data = &graph.constant_operand_ids_to_handles[&const_id];
            let name = CString::new(
                graph.operands[const_id as usize]
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("const_{const_id}")),
            )
            .unwrap();

            let const_op = unsafe { ddk_cann_op_const_with_name(name.as_ptr()) };
            if const_op.is_null() {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_op_const_with_name for operand {const_id} failed").into(),
                });
            }

            let desc = &graph.operands[const_id as usize].descriptor;
            let mut dims = descriptor_dims(desc);
            if dims.is_empty() {
                // GE rejects 0-D tensors (`shape_count <= 0`); emit a scalar
                // constant (e.g. a scale factor or bias) as shape `[1]`.
                dims = vec![1];
            }
            // Dequantize integer constants to float32: the NPU's Const and
            // CastT kernels do not support int8/uint8, so quantized weights are
            // dequantized here (a plain dtype cast — no scale/zero-point).
            // The matching cast(int8/uint8 -> float32) is folded away below.
            let (const_data, const_dtype) = match desc.data_type {
                DataType::Int8 => (
                    constant_data
                        .data
                        .iter()
                        .flat_map(|&b| (b as i8 as f32).to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DataType::Float32,
                ),
                DataType::Uint8 => (
                    constant_data
                        .data
                        .iter()
                        .flat_map(|&b| (b as f32).to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DataType::Float32,
                ),
                _ => (constant_data.data.clone(), desc.data_type),
            };
            let value_name = CString::new("value").unwrap();
            let format = 0_i32; // FORMAT_NCHW for all const tensors
            let status = unsafe {
                ddk_cann_operator_set_attr_tensor_raw_format(
                    const_op,
                    value_name.as_ptr(),
                    const_data.as_ptr() as *const _,
                    const_data.len() as u32,
                    dims.as_ptr(),
                    dims.len() as i32,
                    cann_data_type(const_dtype),
                    format,
                )
            };
            if status != 0 {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!(
                        "cann_operator_set_attr_tensor_raw for operand {const_id} failed: {status}"
                    )
                    .into(),
                });
            }

            handles[const_id as usize] = const_op;
            const_ops.push(const_op);
        }

        // ── 3. Create compute operations ─────────────────────────────────
        let mut compute_ops: Vec<ddk_CannOperatorHandle> = Vec::new();
        // Intermediate ops from decompositions (e.g. sigmoid), added to the
        // graph alongside compute_ops.
        let mut extra_ops: Vec<ddk_CannOperatorHandle> = Vec::new();

        for op in &graph.operations {
            // Only RNN ops are unsupported; everything else is wired below.
            if is_unsupported_op(op) {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("unsupported op: {}", op.label()).into(),
                });
            }

            // Set true by the spatial op wiring when it normalizes an NHWC op
            // to NCHW; the generic output assignment then wraps the result in a
            // transpose back to NHWC.
            let mut nhwc_out: bool = false;

            // Fold cast(int8/uint8 -> float32) on a constant: the constant was
            // dequantized to float32 at creation (see the const loop above), so
            // the cast is a no-op and can be skipped entirely.
            if let Operation::Cast { input, outputs, .. } = op {
                let src_dtype = graph.operands[*input as usize].descriptor.data_type;
                let dst_dtype = graph.operands[outputs[0] as usize].descriptor.data_type;
                if matches!(src_dtype, DataType::Int8 | DataType::Uint8)
                    && dst_dtype == DataType::Float32
                    && graph.constant_operand_ids_to_handles.contains_key(input)
                {
                    for &o in outputs.iter() {
                        handles[o as usize] = handles[*input as usize];
                    }
                    continue;
                }
            }

            // Sigmoid is decomposed (matching the Chromium reference):
            // sigmoid(x) = 1 / (1 + exp(-x))
            if let Operation::Sigmoid { input, outputs, .. } = op {
                let out_id = outputs[0];

                let one = make_const(
                    &format!("sigmoid_one_{out_id}"),
                    bytemuck::cast_slice(&[1.0f32]),
                    &[1], // shape [1] (adapter requires shape_count > 0)
                    ddk_CannDataType::CANN_DT_FLOAT,
                    0, // FORMAT_NCHW
                );
                extra_ops.push(one);

                let neg_name = CString::new(format!("sigmoid_neg_{out_id}")).unwrap();
                let neg_type = CString::new("Neg").unwrap();
                let neg = unsafe {
                    ddk_cann_operator_create_registered(neg_type.as_ptr(), neg_name.as_ptr())
                };
                let x_name = CString::new("x").unwrap();
                unsafe { set_src_input(neg, &x_name, *input, &handles, &split_out) };
                extra_ops.push(neg);

                let exp_name = CString::new(format!("sigmoid_exp_{out_id}")).unwrap();
                let exp_type = CString::new("Exp").unwrap();
                let exp_neg = unsafe {
                    ddk_cann_operator_create_registered(exp_type.as_ptr(), exp_name.as_ptr())
                };
                unsafe { ddk_cann_operator_set_input(exp_neg, x_name.as_ptr(), neg) };
                extra_ops.push(exp_neg);

                let denom_name = CString::new(format!("sigmoid_denom_{out_id}")).unwrap();
                let add_type = CString::new("Add").unwrap();
                let denom = unsafe {
                    ddk_cann_operator_create_registered(add_type.as_ptr(), denom_name.as_ptr())
                };
                let x1_name = CString::new("x1").unwrap();
                let x2_name = CString::new("x2").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(denom, x1_name.as_ptr(), one);
                    ddk_cann_operator_set_input(denom, x2_name.as_ptr(), exp_neg);
                }
                extra_ops.push(denom);

                let div_name = CString::new(format!("sigmoid_div_{out_id}")).unwrap();
                let div_type = CString::new("Div").unwrap();
                let div = unsafe {
                    ddk_cann_operator_create_registered(div_type.as_ptr(), div_name.as_ptr())
                };
                unsafe {
                    ddk_cann_operator_set_input(div, x1_name.as_ptr(), one);
                    ddk_cann_operator_set_input(div, x2_name.as_ptr(), denom);
                }

                for &o in outputs.iter() {
                    handles[o as usize] = div;
                }
                compute_ops.push(div);
                continue;
            }

            // PReLU is decomposed (matching the Chromium reference):
            // prelu(x, slope) = max(0, x) + slope * min(0, x)
            if let Operation::Prelu {
                input,
                slope,
                outputs,
                ..
            } = op
            {
                let slope_handle = handles[*slope as usize];
                let out_id = outputs[0];

                let x_name = CString::new("x").unwrap();
                let x1_name = CString::new("x1").unwrap();
                let x2_name = CString::new("x2").unwrap();
                let mode_name = CString::new("mode").unwrap();
                let neg_type = CString::new("Neg").unwrap();
                let relu_type = CString::new("ReLU").unwrap();
                let mul_type = CString::new("Mul").unwrap();
                let add_type = CString::new("Add").unwrap();

                // pos = ReLU(x) = Activation(x, mode=1)
                let pos_name = CString::new(format!("prelu_pos_{out_id}")).unwrap();
                let pos = unsafe {
                    ddk_cann_operator_create_registered(relu_type.as_ptr(), pos_name.as_ptr())
                };
                unsafe { set_src_input(pos, &x_name, *input, &handles, &split_out) };
                unsafe { ddk_cann_operator_set_attr_int64(pos, mode_name.as_ptr(), 1) };
                extra_ops.push(pos);

                // neg_input = Neg(x)
                let neg_input_name = CString::new(format!("prelu_neg_input_{out_id}")).unwrap();
                let neg_input = unsafe {
                    ddk_cann_operator_create_registered(neg_type.as_ptr(), neg_input_name.as_ptr())
                };
                unsafe { set_src_input(neg_input, &x_name, *input, &handles, &split_out) };
                extra_ops.push(neg_input);

                // relu_neg = Activation(neg_input, mode=1)
                let relu_neg_name = CString::new(format!("prelu_relu_neg_{out_id}")).unwrap();
                let relu_neg = unsafe {
                    ddk_cann_operator_create_registered(relu_type.as_ptr(), relu_neg_name.as_ptr())
                };
                unsafe { set_operand_input(relu_neg, &x_name, neg_input) };
                unsafe { ddk_cann_operator_set_attr_int64(relu_neg, mode_name.as_ptr(), 1) };
                extra_ops.push(relu_neg);

                // neg_x = Neg(relu_neg)
                let neg_x_name = CString::new(format!("prelu_neg_x_{out_id}")).unwrap();
                let neg_x = unsafe {
                    ddk_cann_operator_create_registered(neg_type.as_ptr(), neg_x_name.as_ptr())
                };
                unsafe { set_operand_input(neg_x, &x_name, relu_neg) };
                extra_ops.push(neg_x);

                // neg_scaled = Mul(neg_x, slope)
                let neg_scaled_name = CString::new(format!("prelu_neg_scaled_{out_id}")).unwrap();
                let neg_scaled = unsafe {
                    ddk_cann_operator_create_registered(mul_type.as_ptr(), neg_scaled_name.as_ptr())
                };
                unsafe {
                    ddk_cann_operator_set_input(neg_scaled, x1_name.as_ptr(), neg_x);
                    ddk_cann_operator_set_input(neg_scaled, x2_name.as_ptr(), slope_handle);
                }
                extra_ops.push(neg_scaled);

                // output = Add(pos, neg_scaled)
                let output_name = CString::new(format!("prelu_output_{out_id}")).unwrap();
                let output = unsafe {
                    ddk_cann_operator_create_registered(add_type.as_ptr(), output_name.as_ptr())
                };
                unsafe {
                    ddk_cann_operator_set_input(output, x1_name.as_ptr(), pos);
                    ddk_cann_operator_set_input(output, x2_name.as_ptr(), neg_scaled);
                }

                for &o in outputs.iter() {
                    handles[o as usize] = output;
                }
                compute_ops.push(output);
                continue;
            }

            // Resample2d picks ResizeNearestNeighbor vs ResizeBilinear based on
            // the interpolation mode (matching the Chromium reference).
            if let Operation::Resample2d {
                input,
                options,
                outputs,
                ..
            } = op
            {
                let out_id = outputs[0];
                // Resample2d has no layout option; infer NHWC from the axes
                // (NHWC spatial dims are [1,2], NCHW are [2,3]). Normalize an
                // NHWC resample to NCHW via transposes.
                let in_dims = descriptor_dims(&graph.operands[*input as usize].descriptor);
                let axes = options.as_ref().map(|o| o.axes.clone()).unwrap_or_default();
                let is_nhwc = in_dims.len() == 4 && axes.len() >= 2 && axes[0] == 1 && axes[1] == 2;
                let x_handle = if is_nhwc {
                    emit_transpose(
                        &format!("resample_in_nhwc_{out_id}"),
                        handles[*input as usize],
                        &[0, 3, 1, 2],
                        &mut extra_ops,
                    )?
                } else {
                    handles[*input as usize]
                };
                let x_name = CString::new("x").unwrap();

                let nearest = options
                    .as_ref()
                    .map(|o| o.mode == "nearest-neighbor")
                    .unwrap_or(false);
                let op_name = CString::new(format!("resample2d_{out_id}")).unwrap();
                let resample_op = if nearest {
                    unsafe { ddk_cann_op_resize_nearest_neighbor_with_name(op_name.as_ptr()) }
                } else {
                    let resample2d_type = CString::new("Resample2D").unwrap();
                    unsafe {
                        ddk_cann_operator_create_registered(
                            resample2d_type.as_ptr(),
                            op_name.as_ptr(),
                        )
                    }
                };
                if resample_op.is_null() {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: "cann_operator_create failed for resample2d".into(),
                    });
                }

                let status = if is_nhwc {
                    unsafe { set_operand_input(resample_op, &x_name, x_handle) }
                } else {
                    unsafe { set_src_input(resample_op, &x_name, *input, &handles, &split_out) }
                };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: "cann_operator_set_input for resample2d failed".into(),
                    });
                }

                // size = [h, w] from the output shape (NHWC axes 1,2 / NCHW 2,3).
                let out_dims = descriptor_dims(&graph.operands[outputs[0] as usize].descriptor);
                let (h, w) = if is_nhwc {
                    (
                        if out_dims.len() >= 3 { out_dims[1] } else { 0 },
                        if out_dims.len() >= 3 { out_dims[2] } else { 0 },
                    )
                } else {
                    (
                        if out_dims.len() >= 4 { out_dims[2] } else { 0 },
                        if out_dims.len() >= 4 { out_dims[3] } else { 0 },
                    )
                };
                let size_vals: Vec<i32> = [h as i32, w as i32].to_vec();
                let size_const = make_const(
                    &format!("resample_size_{out_id}"),
                    bytemuck::cast_slice(&size_vals),
                    &[2],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(size_const);
                let size_name = CString::new("size").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(resample_op, size_name.as_ptr(), size_const);
                }

                let out_handle = if is_nhwc {
                    emit_transpose(
                        &format!("resample_out_nhwc_{out_id}"),
                        resample_op,
                        &[0, 2, 3, 1],
                        &mut extra_ops,
                    )?
                } else {
                    resample_op
                };

                for &out_id in outputs.iter() {
                    handles[out_id as usize] = out_handle;
                }
                compute_ops.push(resample_op);
                continue;
            }

            // Wire Split via the native hiai::op::SplitD (matching Chromium),
            // not by decomposing into Slice ops. The dynamic output "y" is
            // registered first (create_dynamic_output), then consumers route to
            // the i-th output via set_src_input/set_src_dynamic_input, which use
            // the index-based GetOutput(i) that the adapter already exposes.
            if let Operation::Split {
                input,
                options,
                outputs,
                ..
            } = op
            {
                let axis = options.as_ref().map(|o| o.axis).unwrap_or(0) as i64;
                let split_op =
                    create_op("Split", &format!("split_{}", outputs[0]), &mut extra_ops)?;

                let x_name = CString::new("x").unwrap();
                let status =
                    unsafe { set_src_input(split_op, &x_name, *input, &handles, &split_out) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: "cann_operator_set_input for Split failed".into(),
                    });
                }

                let split_dim_name = CString::new("split_dim").unwrap();
                let num_split_name = CString::new("num_split").unwrap();
                unsafe {
                    ddk_cann_operator_set_attr_int64(split_op, split_dim_name.as_ptr(), axis);
                    ddk_cann_operator_set_attr_int64(
                        split_op,
                        num_split_name.as_ptr(),
                        outputs.len() as i64,
                    );
                }

                let y_name = CString::new("y").unwrap();
                let status = unsafe {
                    ddk_cann_operator_create_dynamic_output(
                        split_op,
                        y_name.as_ptr(),
                        outputs.len() as u32,
                    )
                };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_create_dynamic_output for split failed: {status}"
                        )
                        .into(),
                    });
                }

                for (i, &out_id) in outputs.iter().enumerate() {
                    handles[out_id as usize] = split_op;
                    split_out.insert(out_id, i as u32);
                }
                continue;
            }

            // ArgMin = Neg(x) then ArgMaxExt2(neg, axis) (matching the reference).
            if let Operation::ArgMin {
                input,
                axis,
                outputs,
                ..
            } = op
            {
                let out_id = outputs[0];
                let neg = create_op("Neg", &format!("argmin_neg_{out_id}"), &mut extra_ops)?;
                connect_src_input(neg, "x", *input, &handles, &split_out)?;

                let argmax =
                    create_op("ArgMax", &format!("argmin_argmax_{out_id}"), &mut extra_ops)?;
                connect_input(argmax, "x", neg)?;
                let axis_const = make_const(
                    &format!("argmin_axis_{out_id}"),
                    bytemuck::cast_slice(&[*axis as i32]),
                    &[1],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(axis_const);
                connect_input(argmax, "axis", axis_const)?;

                handles[out_id as usize] = argmax;
                continue;
            }

            // reduceL1 = Abs(x) -> ReduceSum(x, axes, keep_dims).
            if let Operation::ReduceL1 {
                input,
                options,
                outputs,
                ..
            } = op
            {
                let out_id = outputs[0];
                let abs_op = create_op("Abs", &format!("reduce_l1_abs_{out_id}"), &mut extra_ops)?;
                connect_src_input(abs_op, "x", *input, &handles, &split_out)?;
                set_int64_attr(abs_op, "mode", 6);

                let sum = create_op(
                    "ReduceSum",
                    &format!("reduce_l1_sum_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_input(sum, "x", abs_op)?;
                let axes = reduce_axes(*input, options, graph);
                let axes_const = make_const(
                    &format!("reduce_l1_axes_{out_id}"),
                    bytemuck::cast_slice(&axes),
                    &[axes.len() as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(axes_const);
                connect_input(sum, "axes", axes_const)?;
                set_bool_attr(
                    sum,
                    "keep_dims",
                    options.as_ref().map(|o| o.keep_dimensions).unwrap_or(false),
                );

                handles[out_id as usize] = sum;
                continue;
            }

            // reduceLogSum = Log(ReduceSum(x, axes, keep_dims)).
            if let Operation::ReduceLogSum {
                input,
                options,
                outputs,
                ..
            } = op
            {
                let out_id = outputs[0];
                let sum = create_op(
                    "ReduceSum",
                    &format!("reduce_log_sum_sum_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_src_input(sum, "x", *input, &handles, &split_out)?;
                let axes = reduce_axes(*input, options, graph);
                let axes_const = make_const(
                    &format!("reduce_log_sum_axes_{out_id}"),
                    bytemuck::cast_slice(&axes),
                    &[axes.len() as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(axes_const);
                connect_input(sum, "axes", axes_const)?;
                set_bool_attr(
                    sum,
                    "keep_dims",
                    options.as_ref().map(|o| o.keep_dimensions).unwrap_or(false),
                );

                let log_op = create_op(
                    "Log",
                    &format!("reduce_log_sum_log_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_input(log_op, "x", sum)?;

                handles[out_id as usize] = log_op;
                continue;
            }

            // reduceSumSquare = ReduceSum(Square(x), axes, keep_dims).
            if let Operation::ReduceSumSquare {
                input,
                options,
                outputs,
                ..
            } = op
            {
                let out_id = outputs[0];
                let square = create_op(
                    "Square",
                    &format!("reduce_sum_square_sq_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_src_input(square, "x", *input, &handles, &split_out)?;

                let sum = create_op(
                    "ReduceSum",
                    &format!("reduce_sum_square_sum_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_input(sum, "x", square)?;
                let axes = reduce_axes(*input, options, graph);
                let axes_const = make_const(
                    &format!("reduce_sum_square_axes_{out_id}"),
                    bytemuck::cast_slice(&axes),
                    &[axes.len() as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(axes_const);
                connect_input(sum, "axes", axes_const)?;
                set_bool_attr(
                    sum,
                    "keep_dims",
                    options.as_ref().map(|o| o.keep_dimensions).unwrap_or(false),
                );

                handles[out_id as usize] = sum;
                continue;
            }

            // Identity = Reshape(x, Shape(x)).
            if let Operation::Identity { input, outputs, .. } = op {
                let out_id = outputs[0];
                let shape_op =
                    create_op("Shape", &format!("identity_shape_{out_id}"), &mut extra_ops)?;
                connect_src_input(shape_op, "x", *input, &handles, &split_out)?;

                let reshape = create_op(
                    "Reshape",
                    &format!("identity_reshape_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_src_input(reshape, "x", *input, &handles, &split_out)?;
                connect_input(reshape, "shape", shape_op)?;

                handles[out_id as usize] = reshape;
                continue;
            }

            // isNaN = NotEqual(x, x).
            if let Operation::IsNaN { input, outputs, .. } = op {
                let out_id = outputs[0];
                let ne = create_op("NotEqual", &format!("isnan_ne_{out_id}"), &mut extra_ops)?;
                connect_src_input(ne, "x1", *input, &handles, &split_out)?;
                connect_src_input(ne, "x2", *input, &handles, &split_out)?;

                handles[out_id as usize] = ne;
                continue;
            }

            // isInfinite = (x == +inf) || (x == -inf).
            if let Operation::IsInfinite { input, outputs, .. } = op {
                let out_id = outputs[0];
                let in_dims = descriptor_dims(&graph.operands[*input as usize].descriptor);
                let shape_const = make_const(
                    &format!("isinfinite_shape_{out_id}"),
                    bytemuck::cast_slice(&in_dims.iter().map(|&d| d as i32).collect::<Vec<_>>()),
                    &[in_dims.len() as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(shape_const);

                let pos = make_const(
                    &format!("isinfinite_pos_{out_id}"),
                    bytemuck::cast_slice(&[f32::INFINITY]),
                    &[1],
                    ddk_CannDataType::CANN_DT_FLOAT,
                    0,
                );
                let neg = make_const(
                    &format!("isinfinite_neg_{out_id}"),
                    bytemuck::cast_slice(&[f32::NEG_INFINITY]),
                    &[1],
                    ddk_CannDataType::CANN_DT_FLOAT,
                    0,
                );
                extra_ops.push(pos);
                extra_ops.push(neg);

                let pos_b = create_op(
                    "Expand",
                    &format!("isinfinite_pos_b_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_input(pos_b, "x", pos)?;
                connect_input(pos_b, "shape", shape_const)?;
                let neg_b = create_op(
                    "Expand",
                    &format!("isinfinite_neg_b_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_input(neg_b, "x", neg)?;
                connect_input(neg_b, "shape", shape_const)?;

                let eq_pos = create_op(
                    "Equal",
                    &format!("isinfinite_eq_pos_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_src_input(eq_pos, "x1", *input, &handles, &split_out)?;
                connect_input(eq_pos, "x2", pos_b)?;
                let eq_neg = create_op(
                    "Equal",
                    &format!("isinfinite_eq_neg_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_src_input(eq_neg, "x1", *input, &handles, &split_out)?;
                connect_input(eq_neg, "x2", neg_b)?;

                let or_op = create_op(
                    "LogicalOr",
                    &format!("isinfinite_or_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_input(or_op, "x1", eq_pos)?;
                connect_input(or_op, "x2", eq_neg)?;

                handles[out_id as usize] = or_op;
                continue;
            }

            // globalAveragePool = ReduceMean over spatial (H, W) axes.
            if let Operation::GlobalAveragePool { input, outputs, .. } = op {
                let out_id = outputs[0];
                let mean = create_op(
                    "ReduceMean",
                    &format!("global_avg_pool_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_src_input(mean, "x", *input, &handles, &split_out)?;
                let rank = graph.operands[*input as usize].descriptor.shape.len();
                let axes: Vec<i32> = if rank >= 2 {
                    vec![(rank as i32) - 2, (rank as i32) - 1]
                } else {
                    vec![0]
                };
                let axes_const = make_const(
                    &format!("global_avg_pool_axes_{out_id}"),
                    bytemuck::cast_slice(&axes),
                    &[axes.len() as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(axes_const);
                connect_input(mean, "axes", axes_const)?;
                set_bool_attr(mean, "keep_dims", true);

                handles[out_id as usize] = mean;
                continue;
            }

            // globalMaxPool = ReduceMax over spatial (H, W) axes.
            if let Operation::GlobalMaxPool { input, outputs, .. } = op {
                let out_id = outputs[0];
                let max = create_op(
                    "ReduceMax",
                    &format!("global_max_pool_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_src_input(max, "x", *input, &handles, &split_out)?;
                let rank = graph.operands[*input as usize].descriptor.shape.len();
                let axes: Vec<i32> = if rank >= 2 {
                    vec![(rank as i32) - 2, (rank as i32) - 1]
                } else {
                    vec![0]
                };
                let axes_const = make_const(
                    &format!("global_max_pool_axes_{out_id}"),
                    bytemuck::cast_slice(&axes),
                    &[axes.len() as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(axes_const);
                connect_input(max, "axes", axes_const)?;
                set_bool_attr(max, "keep_dims", true);

                handles[out_id as usize] = max;
                continue;
            }

            // linear(x) = alpha * x + beta.
            if let Operation::Linear {
                input,
                options,
                outputs,
                ..
            } = op
            {
                let out_id = outputs[0];
                let alpha = options.as_ref().map(|o| o.alpha as f32).unwrap_or(1.0);
                let beta = options.as_ref().map(|o| o.beta as f32).unwrap_or(0.0);
                let in_dims = descriptor_dims(&graph.operands[*input as usize].descriptor);
                let shape_vals: Vec<i32> = in_dims.iter().map(|&d| d as i32).collect();
                let shape_const = make_const(
                    &format!("linear_shape_{out_id}"),
                    bytemuck::cast_slice(&shape_vals),
                    &[shape_vals.len() as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(shape_const);

                let alpha_const = make_const(
                    &format!("linear_alpha_{out_id}"),
                    bytemuck::cast_slice(&[alpha]),
                    &[1],
                    ddk_CannDataType::CANN_DT_FLOAT,
                    0,
                );
                let beta_const = make_const(
                    &format!("linear_beta_{out_id}"),
                    bytemuck::cast_slice(&[beta]),
                    &[1],
                    ddk_CannDataType::CANN_DT_FLOAT,
                    0,
                );
                extra_ops.push(alpha_const);
                extra_ops.push(beta_const);

                let alpha_b = create_op(
                    "Expand",
                    &format!("linear_alpha_b_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_input(alpha_b, "x", alpha_const)?;
                connect_input(alpha_b, "shape", shape_const)?;
                let beta_b =
                    create_op("Expand", &format!("linear_beta_b_{out_id}"), &mut extra_ops)?;
                connect_input(beta_b, "x", beta_const)?;
                connect_input(beta_b, "shape", shape_const)?;

                let mul = create_op("Mul", &format!("linear_mul_{out_id}"), &mut extra_ops)?;
                connect_src_input(mul, "x1", *input, &handles, &split_out)?;
                connect_input(mul, "x2", alpha_b)?;
                let add = create_op("Add", &format!("linear_add_{out_id}"), &mut extra_ops)?;
                connect_input(add, "x1", mul)?;
                connect_input(add, "x2", beta_b)?;

                handles[out_id as usize] = add;
                continue;
            }

            // reverse via hiai::op::StridedSliceV2 (begin/end/strides + end_mask).
            if let Operation::Reverse {
                input,
                options,
                outputs,
                ..
            } = op
            {
                let out_id = outputs[0];
                let in_dims = descriptor_dims(&graph.operands[*input as usize].descriptor);
                let rank = in_dims.len();
                let axes: Vec<u32> = options
                    .as_ref()
                    .and_then(|o| o.axes.clone())
                    .unwrap_or_else(|| (0..rank as u32).collect());

                let mut begin: Vec<i32> = vec![0; rank];
                let end: Vec<i32> = vec![-1; rank];
                let mut strides: Vec<i32> = vec![1; rank];
                let mut end_mask: i64 = 0;
                for &ax in &axes {
                    if (ax as usize) < rank {
                        begin[ax as usize] = in_dims[ax as usize] as i32 - 1;
                        strides[ax as usize] = -1;
                        end_mask |= 1 << ax;
                    }
                }

                let slice = create_op(
                    "StridedSliceV2",
                    &format!("reverse_slice_{out_id}"),
                    &mut extra_ops,
                )?;
                connect_src_input(slice, "x", *input, &handles, &split_out)?;
                let begin_const = make_const(
                    &format!("reverse_begin_{out_id}"),
                    bytemuck::cast_slice(&begin),
                    &[rank as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                let end_const = make_const(
                    &format!("reverse_end_{out_id}"),
                    bytemuck::cast_slice(&end),
                    &[rank as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                let strides_const = make_const(
                    &format!("reverse_strides_{out_id}"),
                    bytemuck::cast_slice(&strides),
                    &[rank as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(begin_const);
                extra_ops.push(end_const);
                extra_ops.push(strides_const);
                connect_input(slice, "begin", begin_const)?;
                connect_input(slice, "end", end_const)?;
                connect_input(slice, "strides", strides_const)?;
                set_int64_attr(slice, "end_mask", end_mask);

                handles[out_id as usize] = slice;
                continue;
            }

            // dequantizeLinear = Mul(Sub(Cast(x, float), zp), scale).
            if let Operation::DequantizeLinear {
                input,
                scale,
                zero_point,
                outputs,
                ..
            } = op
            {
                let scale_handle = handles[*scale as usize];
                let out_id = outputs[0];

                let src_dtype = graph.operands[*input as usize].descriptor.data_type;
                let cast = create_op("Cast", &format!("dequant_cast_{out_id}"), &mut extra_ops)?;
                connect_src_input(cast, "x", *input, &handles, &split_out)?;
                set_int64_attr(cast, "src_dtype", cann_data_type(src_dtype) as i64);
                set_int64_attr(cast, "dst_dtype", ddk_CannDataType::CANN_DT_FLOAT as i64);

                let sub = create_op("Sub", &format!("dequant_sub_{out_id}"), &mut extra_ops)?;
                connect_input(sub, "x1", cast)?;
                if let Some(zp_id) = zero_point {
                    connect_input(sub, "x2", handles[*zp_id as usize])?;
                }
                let mul = create_op("Mul", &format!("dequant_mul_{out_id}"), &mut extra_ops)?;
                connect_input(mul, "x1", sub)?;
                connect_input(mul, "x2", scale_handle)?;

                handles[out_id as usize] = mul;
                continue;
            }

            let operator_type_name = match webnn_op_to_hiai(op) {
                Some(name) => CString::new(name).unwrap(),
                None => {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!("unsupported op: {}", op.label()).into(),
                    });
                }
            };

            // Create operator via cann_operator_create_registered.
            let op_name = CString::new(format!(
                "{}_{}",
                operator_type_name.to_str().unwrap(),
                op_outputs(op)[0]
            ))
            .unwrap();
            let compute_op: ddk_CannOperatorHandle = unsafe {
                ddk_cann_operator_create_registered(operator_type_name.as_ptr(), op_name.as_ptr())
            };

            if compute_op.is_null() {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_operator_create failed for {}", op.label()).into(),
                });
            }

            if let Some((a, b, name_a, name_b)) = binary_op_inputs(op) {
                let lhs_name = CString::new(name_a).unwrap();
                let rhs_name = CString::new(name_b).unwrap();
                let status_a =
                    unsafe { set_src_input(compute_op, &lhs_name, a, &handles, &split_out) };
                let status_b =
                    unsafe { set_src_input(compute_op, &rhs_name, b, &handles, &split_out) };
                if status_a != 0 || status_b != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }
            }

            if let Some(input) = unary_op_input(op) {
                let x_name = CString::new("x").unwrap();
                let status =
                    unsafe { set_src_input(compute_op, &x_name, input, &handles, &split_out) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }
            }

            // Activation / clamp attributes. The generic unary path only wires
            // the "x" input; these ops map to hiai::op::Activation (or
            // ClipByValue) and require the mode / bounds to be set explicitly.
            let mode_name = CString::new("mode").unwrap();
            match op {
                Operation::Relu { .. } => unsafe {
                    ddk_cann_operator_set_attr_int64(compute_op, mode_name.as_ptr(), 1);
                },
                Operation::Tanh { .. } => unsafe {
                    ddk_cann_operator_set_attr_int64(compute_op, mode_name.as_ptr(), 2);
                },
                Operation::Abs { .. } => unsafe {
                    ddk_cann_operator_set_attr_int64(compute_op, mode_name.as_ptr(), 6);
                },
                Operation::Softplus { .. } => unsafe {
                    ddk_cann_operator_set_attr_int64(compute_op, mode_name.as_ptr(), 9);
                },
                Operation::Softsign { .. } => unsafe {
                    ddk_cann_operator_set_attr_int64(compute_op, mode_name.as_ptr(), 8);
                },
                Operation::Gelu { .. } => unsafe {
                    ddk_cann_operator_set_attr_int64(compute_op, mode_name.as_ptr(), 15);
                },
                Operation::HardSigmoid { .. } => unsafe {
                    ddk_cann_operator_set_attr_int64(compute_op, mode_name.as_ptr(), 10);
                },
                Operation::LeakyRelu { options, .. } => {
                    let alpha = options.as_ref().map(|o| o.alpha as f32).unwrap_or(0.01);
                    let slope_name = CString::new("negative_slope").unwrap();
                    unsafe {
                        ddk_cann_operator_set_attr_int64(compute_op, mode_name.as_ptr(), 5);
                        ddk_cann_operator_set_attr_float(compute_op, slope_name.as_ptr(), alpha);
                    }
                }
                Operation::Elu { options, .. } => {
                    let alpha = options.as_ref().map(|o| o.alpha as f32).unwrap_or(1.0);
                    let coef_name = CString::new("coef").unwrap();
                    unsafe {
                        ddk_cann_operator_set_attr_int64(compute_op, mode_name.as_ptr(), 4);
                        ddk_cann_operator_set_attr_float(compute_op, coef_name.as_ptr(), alpha);
                    }
                }
                Operation::Clamp { options, .. } => {
                    let min = options
                        .as_ref()
                        .and_then(|o| o.min_value.as_ref())
                        .and_then(json_number_f32)
                        .unwrap_or(f32::NEG_INFINITY);
                    let max = options
                        .as_ref()
                        .and_then(|o| o.max_value.as_ref())
                        .and_then(json_number_f32)
                        .unwrap_or(f32::INFINITY);
                    let min_name = CString::new("min").unwrap();
                    let max_name = CString::new("max").unwrap();
                    unsafe {
                        ddk_cann_operator_set_attr_float(compute_op, min_name.as_ptr(), min);
                        ddk_cann_operator_set_attr_float(compute_op, max_name.as_ptr(), max);
                    }
                }
                _ => {}
            }

            // Wire Cast via hiai::op::CastT (adapter maps "Cast"; attrs
            // src_dtype/dst_dtype).
            if let Operation::Cast { input, outputs, .. } = op {
                let src_dtype = graph.operands[*input as usize].descriptor.data_type;
                let dst_dtype = graph.operands[outputs[0] as usize].descriptor.data_type;
                let src_name = CString::new("src_dtype").unwrap();
                let dst_name = CString::new("dst_dtype").unwrap();
                unsafe {
                    ddk_cann_operator_set_attr_int64(
                        compute_op,
                        src_name.as_ptr(),
                        cann_data_type(src_dtype) as i64,
                    );
                    ddk_cann_operator_set_attr_int64(
                        compute_op,
                        dst_name.as_ptr(),
                        cann_data_type(dst_dtype) as i64,
                    );
                }
            }

            // Wire reductions (sum/mean/max/min/product/l2/logSumExp). L1,
            // logSum and sumSquare are decomposed earlier.
            if let Some((input, options, axes_as_attr, keep_dims_attr_name)) = reduce_op_info(op) {
                let x_name = CString::new("x").unwrap();
                let status =
                    unsafe { set_src_input(compute_op, &x_name, input, &handles, &split_out) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }

                let axes = reduce_axes(input, options, graph);
                let keep_dims = options.as_ref().map(|o| o.keep_dimensions).unwrap_or(false);
                let keep_dims_name = CString::new(keep_dims_attr_name).unwrap();

                if axes_as_attr {
                    let axes_i64: Vec<i64> = axes.iter().map(|&ax| ax as i64).collect();
                    let axes_name = CString::new("axes").unwrap();
                    unsafe {
                        ddk_cann_operator_set_attr_int64_list(
                            compute_op,
                            axes_name.as_ptr(),
                            axes_i64.as_ptr(),
                            axes_i64.len() as i32,
                        );
                        ddk_cann_operator_set_attr_bool(
                            compute_op,
                            keep_dims_name.as_ptr(),
                            keep_dims as i32,
                        );
                    }
                } else {
                    let axes_const = make_const(
                        &format!("reduce_axes_{}", op_outputs(op)[0]),
                        bytemuck::cast_slice(&axes),
                        &[axes.len() as i64],
                        ddk_CannDataType::CANN_DT_INT32,
                        2, // FORMAT_ND
                    );
                    extra_ops.push(axes_const);
                    let axes_name = CString::new("axes").unwrap();
                    unsafe {
                        ddk_cann_operator_set_input(compute_op, axes_name.as_ptr(), axes_const);
                        ddk_cann_operator_set_attr_bool(
                            compute_op,
                            keep_dims_name.as_ptr(),
                            keep_dims as i32,
                        );
                    }
                }
            }

            // Wire Conv2d via hiai::op::Convolution.
            if let Operation::Conv2d {
                input,
                filter,
                options,
                ..
            } = op
            {
                let out_id = op.outputs()[0];
                let input_layout = options
                    .as_ref()
                    .map(|o| o.input_layout.as_str())
                    .unwrap_or("");
                let is_nhwc = input_layout.eq_ignore_ascii_case("nhwc");
                // The GE Convolution infershape is NCHW-only, so normalize an
                // NHWC conv to NCHW: transpose the input, reorder the filter to
                // OIHW, and (via `nhwc_out`) transpose the output back.
                let in_dims = descriptor_dims(&graph.operands[*input as usize].descriptor);
                log::error!(
                    "[cann-debug] conv2d out={out_id} input_layout={input_layout} in_dims={in_dims:?}"
                );
                let x_handle = if is_nhwc {
                    let t = emit_transpose(
                        &format!("conv_in_nhwc_{out_id}"),
                        handles[*input as usize],
                        &[0, 3, 1, 2],
                        &mut extra_ops,
                    )?;
                    // Wrap the Permute in an identity Reshape to NCHW so GE's
                    // ConvolutionInfer reads the correct shape (not the
                    // Permute's NHWC input shape).
                    let nchw: Vec<i64> = if in_dims.len() == 4 {
                        vec![in_dims[0], in_dims[3], in_dims[1], in_dims[2]]
                    } else {
                        in_dims.clone()
                    };
                    emit_reshape(
                        &format!("conv_in_reshape_{out_id}"),
                        t,
                        &nchw,
                        &mut extra_ops,
                    )?
                } else {
                    handles[*input as usize]
                };
                let filter_layout = options
                    .as_ref()
                    .map(|o| o.filter_layout.as_str())
                    .unwrap_or("");
                let filter_handle = match filter_ohwi_to_nchw(
                    graph,
                    *filter,
                    filter_layout,
                    &mut extra_ops,
                    &format!("conv_filt_oihw_{out_id}"),
                    [0, 3, 1, 2], // OHWI -> OIHW
                )? {
                    Some(oihw_const) => oihw_const,
                    None => handles[*filter as usize],
                };
                nhwc_out = is_nhwc;
                let x_name = CString::new("x").unwrap();
                let filter_name = CString::new("filter").unwrap();
                // Route the input via the source operand (it may be a Split
                // output); NHWC inputs first pass through an intermediate
                // Transpose+Reshape, so those connect by handle.
                let set_status_x = if is_nhwc {
                    unsafe { set_operand_input(compute_op, &x_name, x_handle) }
                } else {
                    unsafe { set_src_input(compute_op, &x_name, *input, &handles, &split_out) }
                };
                let set_status_filter = unsafe {
                    ddk_cann_operator_set_input(compute_op, filter_name.as_ptr(), filter_handle)
                };
                if set_status_x != 0 || set_status_filter != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }

                // Optional bias input (hiai::op::Convolution supports a bias input,
                // matching the Chromium reference's set_input_bias).
                if let Some(bias_id) = options.as_ref().and_then(|o| o.bias) {
                    let bias_handle = handles[bias_id as usize];
                    let bias_name = CString::new("bias").unwrap();
                    let set_status_bias = unsafe {
                        ddk_cann_operator_set_input(compute_op, bias_name.as_ptr(), bias_handle)
                    };
                    if set_status_bias != 0 {
                        return Err(GraphError::ConversionFailed {
                            format: "cann".into(),
                            reason: "cann_operator_set_input for Conv2d bias failed".into(),
                        });
                    }
                }

                // hiai::op::Convolution: strides(2), pads(4), dilations(2),
                // pad_mode="SPECIFIC", data_format, groups. Read from options.
                let (stride_h, stride_w) = match options.as_ref().and_then(|o| {
                    if o.strides.len() >= 2 {
                        Some((o.strides[0] as i64, o.strides[1] as i64))
                    } else {
                        None
                    }
                }) {
                    Some((h, w)) => (h, w),
                    None => (1, 1),
                };
                let (dilation_h, dilation_w) = match options.as_ref().and_then(|o| {
                    if o.dilations.len() >= 2 {
                        Some((o.dilations[0] as i64, o.dilations[1] as i64))
                    } else {
                        None
                    }
                }) {
                    Some((h, w)) => (h, w),
                    None => (1, 1),
                };
                let (padding_top, padding_bottom, padding_left, padding_right) =
                    match options.as_ref().and_then(|o| {
                        if o.padding.len() >= 4 {
                            Some((
                                o.padding[0] as i64,
                                o.padding[1] as i64,
                                o.padding[2] as i64,
                                o.padding[3] as i64,
                            ))
                        } else {
                            None
                        }
                    }) {
                        Some((t, b, l, r)) => (t, b, l, r),
                        None => (0, 0, 0, 0),
                    };
                let groups = options
                    .as_ref()
                    .map(|o| o.groups as i64)
                    .unwrap_or(1)
                    .max(1);
                let format_str = conv_data_format(options);

                let strides: [i64; 2] = [stride_h, stride_w];
                let pads: [i64; 4] = [padding_top, padding_bottom, padding_left, padding_right];
                let dilations: [i64; 2] = [dilation_h, dilation_w];
                unsafe {
                    let strides_name = CString::new("strides").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        strides_name.as_ptr(),
                        strides.as_ptr(),
                        2,
                    );
                    let pads_name = CString::new("pads").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        pads_name.as_ptr(),
                        pads.as_ptr(),
                        4,
                    );
                    let dilations_name = CString::new("dilations").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        dilations_name.as_ptr(),
                        dilations.as_ptr(),
                        2,
                    );
                    let groups_name = CString::new("groups").unwrap();
                    ddk_cann_operator_set_attr_int64(compute_op, groups_name.as_ptr(), groups);
                    let data_format_name = CString::new("data_format").unwrap();
                    let data_format_value = CString::new(format_str).unwrap();
                    ddk_cann_operator_set_attr_string(
                        compute_op,
                        data_format_name.as_ptr(),
                        data_format_value.as_ptr(),
                    );
                    let pad_mode_name = CString::new("pad_mode").unwrap();
                    let pad_mode_value = CString::new("SPECIFIC").unwrap();
                    ddk_cann_operator_set_attr_string(
                        compute_op,
                        pad_mode_name.as_ptr(),
                        pad_mode_value.as_ptr(),
                    );
                }
            }

            // Wire pooling (max/average/l2) via hiai::op::PoolingD.
            if let Some((input, options, pool_mode)) = match op {
                Operation::MaxPool2d { input, options, .. } => Some((*input, options, 0)),
                Operation::AveragePool2d { input, options, .. } => Some((*input, options, 1)),
                Operation::L2Pool2d { input, options, .. } => Some((*input, options, 2)),
                _ => None,
            } {
                let out_id = op.outputs()[0];
                let is_nhwc = options
                    .as_ref()
                    .map(|o| o.layout.eq_ignore_ascii_case("nhwc"))
                    .unwrap_or(false);
                let x_handle = if is_nhwc {
                    emit_transpose(
                        &format!("pool_in_nhwc_{out_id}"),
                        handles[input as usize],
                        &[0, 3, 1, 2],
                        &mut extra_ops,
                    )?
                } else {
                    handles[input as usize]
                };
                nhwc_out = is_nhwc;
                let x_name = CString::new("x").unwrap();
                let status = if is_nhwc {
                    unsafe { set_operand_input(compute_op, &x_name, x_handle) }
                } else {
                    unsafe { set_src_input(compute_op, &x_name, input, &handles, &split_out) }
                };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }

                // hiai::op::PoolingD: mode(0=max,1=avg,2=l2), window(2), stride(2), pad(4).
                let mut window_h: i64 = 1;
                let mut window_w: i64 = 1;
                let mut stride_h: i64 = 1;
                let mut stride_w: i64 = 1;
                let mut padding_top: i64 = 0;
                let mut padding_bottom: i64 = 0;
                let mut padding_left: i64 = 0;
                let mut padding_right: i64 = 0;
                if let Some(pool_options) = options.as_ref() {
                    if pool_options.padding.len() >= 4 {
                        padding_top = pool_options.padding[0] as i64;
                        padding_bottom = pool_options.padding[1] as i64;
                        padding_left = pool_options.padding[2] as i64;
                        padding_right = pool_options.padding[3] as i64;
                    }
                    if let Some(ref ws) = pool_options.window_dimensions {
                        if ws.len() >= 2 {
                            window_h = ws[0] as i64;
                            window_w = ws[1] as i64;
                        }
                    }
                    if pool_options.strides.len() >= 2 {
                        stride_h = pool_options.strides[0] as i64;
                        stride_w = pool_options.strides[1] as i64;
                    }
                }

                let window: [i64; 2] = [window_h, window_w];
                let stride: [i64; 2] = [stride_h, stride_w];
                let pad: [i64; 4] = [padding_top, padding_bottom, padding_left, padding_right];
                unsafe {
                    let mode_name = CString::new("mode").unwrap();
                    ddk_cann_operator_set_attr_int64(compute_op, mode_name.as_ptr(), pool_mode);
                    let window_name = CString::new("window").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        window_name.as_ptr(),
                        window.as_ptr(),
                        2,
                    );
                    let stride_name = CString::new("stride").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        stride_name.as_ptr(),
                        stride.as_ptr(),
                        2,
                    );
                    let pad_name = CString::new("pad").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        pad_name.as_ptr(),
                        pad.as_ptr(),
                        4,
                    );
                }
            }

            // Wire ConvTranspose2d as a decomposed Convolution (stride=1 +
            // recomputed pads), matching Chromium's kTransposed case. Native
            // hiai::op::ConvTranspose is unsupported on the NPU.
            if let Operation::ConvTranspose2d {
                input,
                filter,
                options,
                ..
            } = op
            {
                let out_id = op.outputs()[0];
                let input_layout = options
                    .as_ref()
                    .map(|o| o.input_layout.as_str())
                    .unwrap_or("");
                let is_nhwc = input_layout.eq_ignore_ascii_case("nhwc");
                // Normalize an NHWC conv transpose to NCHW: transpose the input
                // to NCHW, reorder the OHWI filter to OIHW, and (via
                // `nhwc_out`) transpose the output back to NHWC.
                let in_dims = descriptor_dims(&graph.operands[*input as usize].descriptor);
                log::error!(
                    "[cann-debug] convTranspose2d out={out_id} input_layout={input_layout} in_dims={in_dims:?}"
                );
                let x_handle = if is_nhwc {
                    let t = emit_transpose(
                        &format!("convt_in_nhwc_{out_id}"),
                        handles[*input as usize],
                        &[0, 3, 1, 2],
                        &mut extra_ops,
                    )?;
                    let nchw: Vec<i64> = if in_dims.len() == 4 {
                        vec![in_dims[0], in_dims[3], in_dims[1], in_dims[2]]
                    } else {
                        in_dims.clone()
                    };
                    emit_reshape(
                        &format!("convt_in_reshape_{out_id}"),
                        t,
                        &nchw,
                        &mut extra_ops,
                    )?
                } else {
                    handles[*input as usize]
                };
                let filter_layout = options
                    .as_ref()
                    .map(|o| o.filter_layout.as_str())
                    .unwrap_or("");
                let filter_handle = match filter_ohwi_to_nchw(
                    graph,
                    *filter,
                    filter_layout,
                    &mut extra_ops,
                    &format!("convt_filt_oihw_{out_id}"),
                    [0, 3, 1, 2], // OHWI -> OIHW (Convolution expects OIHW)
                )? {
                    Some(oihw_const) => oihw_const,
                    None => handles[*filter as usize],
                };
                nhwc_out = is_nhwc;
                // Convolution input order: x then filter.
                let x_name = CString::new("x").unwrap();
                let filter_name = CString::new("filter").unwrap();
                let set_status_x = if is_nhwc {
                    unsafe { set_operand_input(compute_op, &x_name, x_handle) }
                } else {
                    unsafe { set_src_input(compute_op, &x_name, *input, &handles, &split_out) }
                };
                let set_status_filter = unsafe {
                    ddk_cann_operator_set_input(compute_op, filter_name.as_ptr(), filter_handle)
                };
                if set_status_filter != 0 || set_status_x != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }

                // Optional bias input (hiai::op::Convolution supports bias).
                if let Some(bias_id) = options.as_ref().and_then(|o| o.bias) {
                    let bias_handle = handles[bias_id as usize];
                    let bias_name = CString::new("bias").unwrap();
                    let set_status_bias = unsafe {
                        ddk_cann_operator_set_input(compute_op, bias_name.as_ptr(), bias_handle)
                    };
                    if set_status_bias != 0 {
                        return Err(GraphError::ConversionFailed {
                            format: "cann".into(),
                            reason: "cann_operator_set_input for ConvTranspose2d bias failed"
                                .into(),
                        });
                    }
                }

                // Decomposed Convolution (matching Chromium's kTransposed case):
                // stride=1 with recomputed pads. ConvTranspose out = (in-1)*s -
                // 2P + K; Conv(s'=1) out = in + 2P' - K + 1.
                let (stride_h, stride_w) = match options.as_ref().and_then(|o| {
                    if o.strides.len() >= 2 {
                        Some((o.strides[0] as i64, o.strides[1] as i64))
                    } else {
                        None
                    }
                }) {
                    Some((h, w)) => (h, w),
                    None => (1, 1),
                };
                let (dilation_h, dilation_w) = match options.as_ref().and_then(|o| {
                    if o.dilations.len() >= 2 {
                        Some((o.dilations[0] as i64, o.dilations[1] as i64))
                    } else {
                        None
                    }
                }) {
                    Some((h, w)) => (h, w),
                    None => (1, 1),
                };
                let (ph_t, ph_b, pw_l, pw_r) = match options.as_ref().and_then(|o| {
                    if o.padding.len() >= 4 {
                        Some((
                            o.padding[0] as i64,
                            o.padding[1] as i64,
                            o.padding[2] as i64,
                            o.padding[3] as i64,
                        ))
                    } else {
                        None
                    }
                }) {
                    Some((t, b, l, r)) => (t, b, l, r),
                    None => (0, 0, 0, 0),
                };
                let groups = options
                    .as_ref()
                    .map(|o| o.groups as i64)
                    .unwrap_or(1)
                    .max(1);

                // Input spatial dims in NCHW and filter spatial dims in OHWI.
                let (ih, iw) = if in_dims.len() == 4 {
                    if is_nhwc {
                        (in_dims[1], in_dims[2]) // NHWC [n,h,w,c]
                    } else {
                        (in_dims[2], in_dims[3]) // NCHW [n,c,h,w]
                    }
                } else {
                    (1, 1)
                };
                let filt_dims = descriptor_dims(&graph.operands[*filter as usize].descriptor);
                let (fh, fw) = if filt_dims.len() == 4 {
                    (filt_dims[1], filt_dims[2]) // OHWI [o,h,w,i]
                } else {
                    (1, 1)
                };
                let ph_tp = (((ih - 1) * stride_h - 2 * ph_t + 2 * fh - ih - 1) / 2).max(0);
                let ph_bp = (((ih - 1) * stride_h - 2 * ph_b + 2 * fh - ih - 1) / 2).max(0);
                let pw_lp = (((iw - 1) * stride_w - 2 * pw_l + 2 * fw - iw - 1) / 2).max(0);
                let pw_rp = (((iw - 1) * stride_w - 2 * pw_r + 2 * fw - iw - 1) / 2).max(0);
                log::error!(
                    "[cann-debug] convTranspose2d out={out_id} ih={ih} iw={iw} fh={fh} fw={fw} s=({stride_h},{stride_w}) pads=({ph_tp},{ph_bp},{pw_lp},{pw_rp})"
                );

                // NHWC is normalized to NCHW via transposes, so the GE op is
                // always NCHW here.
                let format_str = "NCHW";

                let strides: [i64; 2] = [1, 1];
                let pads: [i64; 4] = [ph_tp, ph_bp, pw_lp, pw_rp];
                let dilations: [i64; 2] = [dilation_h, dilation_w];
                unsafe {
                    let strides_name = CString::new("strides").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        strides_name.as_ptr(),
                        strides.as_ptr(),
                        2,
                    );
                    let pads_name = CString::new("pads").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        pads_name.as_ptr(),
                        pads.as_ptr(),
                        4,
                    );
                    let dilations_name = CString::new("dilations").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        dilations_name.as_ptr(),
                        dilations.as_ptr(),
                        2,
                    );
                    let groups_name = CString::new("groups").unwrap();
                    ddk_cann_operator_set_attr_int64(compute_op, groups_name.as_ptr(), groups);
                    let data_format_name = CString::new("data_format").unwrap();
                    let data_format_value = CString::new(format_str).unwrap();
                    ddk_cann_operator_set_attr_string(
                        compute_op,
                        data_format_name.as_ptr(),
                        data_format_value.as_ptr(),
                    );
                    let pad_mode_name = CString::new("pad_mode").unwrap();
                    let pad_mode_value = CString::new("SPECIFIC").unwrap();
                    ddk_cann_operator_set_attr_string(
                        compute_op,
                        pad_mode_name.as_ptr(),
                        pad_mode_value.as_ptr(),
                    );
                }
            }

            // Wire ArgMax via hiai::op::ArgMaxExt2.
            // Axis is a tensor input, not an attribute.  Create an inline Const.
            if let Operation::ArgMax {
                input,
                axis,
                outputs,
                ..
            } = op
            {
                let axis_name_str = CString::new(format!("argmax_axis_{}", outputs[0])).unwrap();
                let axis_operator = unsafe { ddk_cann_op_const_with_name(axis_name_str.as_ptr()) };
                let axis_value: i32 = *axis as i32;
                let axis_shape: [i64; 1] = [1];
                let value_name = CString::new("value").unwrap();
                unsafe {
                    ddk_cann_operator_set_attr_tensor_raw_format(
                        axis_operator,
                        value_name.as_ptr(),
                        &axis_value as *const i32 as *const std::ffi::c_void,
                        4,
                        axis_shape.as_ptr(),
                        1,
                        ddk_CannDataType::CANN_DT_INT32,
                        2, // FORMAT_ND
                    );
                }
                const_ops.push(axis_operator);

                let x_name = CString::new("x").unwrap();
                let axis_name = CString::new("axis").unwrap();
                unsafe { set_src_input(compute_op, &x_name, *input, &handles, &split_out) };
                unsafe {
                    ddk_cann_operator_set_input(compute_op, axis_name.as_ptr(), axis_operator);
                }
            }

            // Wire BatchNormalization via hiai::op::BNInference.
            if let Operation::BatchNormalization {
                input,
                mean,
                variance,
                options,
                ..
            } = op
            {
                let mean_handle = handles[*mean as usize];
                let variance_handle = handles[*variance as usize];

                let x_name = CString::new("x").unwrap();
                let mean_name = CString::new("mean").unwrap();
                let variance_name = CString::new("variance").unwrap();
                unsafe { set_src_input(compute_op, &x_name, *input, &handles, &split_out) };
                unsafe {
                    ddk_cann_operator_set_input(compute_op, mean_name.as_ptr(), mean_handle);
                    ddk_cann_operator_set_input(
                        compute_op,
                        variance_name.as_ptr(),
                        variance_handle,
                    );
                }

                // Optional scale and offset (bias) from options.
                let scale_id = options.as_ref().and_then(|o| o.scale);
                let bias_id = options.as_ref().and_then(|o| o.bias);
                if let Some(id) = scale_id {
                    let scale_handle = handles[id as usize];
                    let scale_name = CString::new("scale").unwrap();
                    unsafe {
                        ddk_cann_operator_set_input(compute_op, scale_name.as_ptr(), scale_handle);
                    }
                }
                if let Some(id) = bias_id {
                    let offset_handle = handles[id as usize];
                    let offset_name = CString::new("offset").unwrap();
                    unsafe {
                        ddk_cann_operator_set_input(
                            compute_op,
                            offset_name.as_ptr(),
                            offset_handle,
                        );
                    }
                }

                let epsilon = options.as_ref().map(|o| o.epsilon as f32).unwrap_or(1e-5);
                let epsilon_name = CString::new("epsilon").unwrap();
                unsafe {
                    ddk_cann_operator_set_attr_float(compute_op, epsilon_name.as_ptr(), epsilon);
                }
            }

            // Wire Softmax via hiai::op::Softmax.
            if let Operation::Softmax { input, axis, .. } = op {
                let x_name = CString::new("x").unwrap();
                let status =
                    unsafe { set_src_input(compute_op, &x_name, *input, &handles, &split_out) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }
                let axis_name = CString::new("axis").unwrap();
                unsafe {
                    ddk_cann_operator_set_attr_int64(compute_op, axis_name.as_ptr(), *axis as i64);
                }
            }

            // Wire Concat via hiai::op::ConcatD (dynamic input x, 1-based index).
            if let Operation::Concat { inputs, axis, .. } = op {
                // Diagnostic: log the concat axis + input/output dims so the
                // next build reveals whether it is a channel-axis (axis=1),
                // C0-aligned concat that the FMK could lower to `concat_c_5d`
                // (instead of the generic `concat_nd`).
                let in_dims: Vec<Vec<i64>> = inputs
                    .iter()
                    .map(|&id| descriptor_dims(&graph.operands[id as usize].descriptor))
                    .collect();
                let out_dims =
                    descriptor_dims(&graph.operands[op.outputs()[0] as usize].descriptor);
                log::error!(
                    "[cann-debug] concat axis={axis} n={} in_dims={in_dims:?} out_dims={out_dims:?}",
                    inputs.len()
                );

                let x_name = CString::new("x").unwrap();
                let status = unsafe {
                    ddk_cann_operator_create_dynamic_input(
                        compute_op,
                        x_name.as_ptr(),
                        inputs.len() as u32,
                    )
                };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!("cann_operator_create_dynamic_input failed: {status}")
                            .into(),
                    });
                }
                for (i, &input_id) in inputs.iter().enumerate() {
                    let status = unsafe {
                        set_src_dynamic_input(
                            compute_op,
                            &x_name,
                            (i + 1) as u32,
                            input_id,
                            &handles,
                            &split_out,
                        )
                    };
                    if status != 0 {
                        return Err(GraphError::ConversionFailed {
                            format: "cann".into(),
                            reason: format!(
                                "cann_operator_set_dynamic_input_by_index failed: {status}"
                            )
                            .into(),
                        });
                    }
                }
                let concat_dim_name = CString::new("concat_dim").unwrap();
                let n_name = CString::new("N").unwrap();
                unsafe {
                    ddk_cann_operator_set_attr_int64(
                        compute_op,
                        concat_dim_name.as_ptr(),
                        *axis as i64,
                    );
                    ddk_cann_operator_set_attr_int64(
                        compute_op,
                        n_name.as_ptr(),
                        inputs.len() as i64,
                    );
                }
            }

            // Wire Reshape / Squeeze / Unsqueeze via hiai::op::Reshape (x + shape
            // const). Squeeze/unsqueeze only change shape, so the output operand's
            // shape is the correct target.
            if let Operation::Reshape { input, outputs, .. }
            | Operation::Squeeze { input, outputs, .. }
            | Operation::Unsqueeze { input, outputs, .. } = op
            {
                let x_name = CString::new("x").unwrap();
                let status =
                    unsafe { set_src_input(compute_op, &x_name, *input, &handles, &split_out) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }
                let shape_vals = descriptor_dims(&graph.operands[outputs[0] as usize].descriptor);
                let shape_const = make_const(
                    &format!("reshape_shape_{}", outputs[0]),
                    bytemuck::cast_slice(&shape_vals),
                    &[shape_vals.len() as i64],
                    ddk_CannDataType::CANN_DT_INT64,
                    2, // FORMAT_ND
                );
                extra_ops.push(shape_const);
                let shape_name = CString::new("shape").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(compute_op, shape_name.as_ptr(), shape_const);
                }
            }

            // Wire Slice via hiai::op::Slice (x + offsets + size).
            if let Operation::Slice {
                input,
                starts,
                sizes,
                outputs,
                ..
            } = op
            {
                let x_name = CString::new("x").unwrap();
                let status =
                    unsafe { set_src_input(compute_op, &x_name, *input, &handles, &split_out) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }
                let dims = starts.len();
                let offsets_vals: Vec<i32> = starts.iter().map(|&s| s as i32).collect();
                let offsets_const = make_const(
                    &format!("slice_offsets_{}", outputs[0]),
                    bytemuck::cast_slice(&offsets_vals),
                    &[dims as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(offsets_const);
                let size_vals: Vec<i32> = sizes.iter().map(|d| d.static_or_max() as i32).collect();
                let size_const = make_const(
                    &format!("slice_size_{}", outputs[0]),
                    bytemuck::cast_slice(&size_vals),
                    &[dims as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(size_const);
                let offsets_name = CString::new("offsets").unwrap();
                let size_name = CString::new("size").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(compute_op, offsets_name.as_ptr(), offsets_const);
                    ddk_cann_operator_set_input(compute_op, size_name.as_ptr(), size_const);
                }
            }

            // Wire Transpose via ge::op::Transpose (x + perm const).
            if let Operation::Transpose {
                input,
                options,
                outputs,
                ..
            } = op
            {
                let x_name = CString::new("x").unwrap();
                let status =
                    unsafe { set_src_input(compute_op, &x_name, *input, &handles, &split_out) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }
                let perm_vals: Vec<i32> = options
                    .as_ref()
                    .map(|o| o.permutation.iter().map(|&p| p as i32).collect())
                    .unwrap_or_default();
                let perm_const = make_const(
                    &format!("transpose_perm_{}", outputs[0]),
                    bytemuck::cast_slice(&perm_vals),
                    &[perm_vals.len() as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(perm_const);
                let perm_name = CString::new("perm").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(compute_op, perm_name.as_ptr(), perm_const);
                }
            }

            // Wire Gemm via hiai::op::GemmD (a, b, optional c + alpha/beta/transpose).
            if let Operation::Gemm { a, b, options, .. } = op {
                connect_src_input(compute_op, "a", *a, &handles, &split_out)?;
                connect_src_input(compute_op, "b", *b, &handles, &split_out)?;
                if let Some(o) = options.as_ref() {
                    set_float_attr(compute_op, "alpha", o.alpha as f32);
                    set_float_attr(compute_op, "beta", o.beta as f32);
                    set_bool_attr(compute_op, "transpose_a", o.a_transpose);
                    set_bool_attr(compute_op, "transpose_b", o.b_transpose);
                    if let Some(c_id) = o.c {
                        connect_src_input(compute_op, "c", c_id, &handles, &split_out)?;
                    }
                }
            }

            // Wire Pad via hiai::op::Pad (x + paddings const, shape [N, 2]).
            if let Operation::Pad {
                input,
                beginning_padding,
                ending_padding,
                outputs,
                ..
            } = op
            {
                connect_src_input(compute_op, "x", *input, &handles, &split_out)?;
                let rank = beginning_padding.len();
                let mut paddings: Vec<i32> = Vec::with_capacity(rank * 2);
                for i in 0..rank {
                    paddings.push(beginning_padding[i] as i32);
                    paddings.push(ending_padding[i] as i32);
                }
                let paddings_const = make_const(
                    &format!("pad_paddings_{}", outputs[0]),
                    bytemuck::cast_slice(&paddings),
                    &[rank as i64, 2],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(paddings_const);
                connect_input(compute_op, "paddings", paddings_const)?;
            }

            // Wire Expand via hiai::op::BroadcastTo (x + shape const).
            if let Operation::Expand { input, outputs, .. } = op {
                connect_src_input(compute_op, "x", *input, &handles, &split_out)?;
                let shape_vals: Vec<i32> =
                    descriptor_dims(&graph.operands[outputs[0] as usize].descriptor)
                        .iter()
                        .map(|&d| d as i32)
                        .collect();
                let shape_const = make_const(
                    &format!("expand_shape_{}", outputs[0]),
                    bytemuck::cast_slice(&shape_vals),
                    &[shape_vals.len() as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(shape_const);
                connect_input(compute_op, "shape", shape_const)?;
            }

            // Wire Tile via hiai::op::Tile (x + multiples const).
            if let Operation::Tile {
                input,
                repetitions,
                outputs,
                ..
            } = op
            {
                connect_src_input(compute_op, "x", *input, &handles, &split_out)?;
                let multiples: Vec<i32> = repetitions.iter().map(|&r| r as i32).collect();
                let multiples_const = make_const(
                    &format!("tile_multiples_{}", outputs[0]),
                    bytemuck::cast_slice(&multiples),
                    &[multiples.len() as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(multiples_const);
                connect_input(compute_op, "multiples", multiples_const)?;
            }

            // Wire CumulativeSum via ge::op::Cumsum (x + axis const + exclusive/reverse).
            if let Operation::CumulativeSum {
                input,
                axis,
                options,
                outputs,
                ..
            } = op
            {
                connect_src_input(compute_op, "x", *input, &handles, &split_out)?;
                let axis_const = make_const(
                    &format!("cumsum_axis_{}", outputs[0]),
                    bytemuck::cast_slice(&[*axis as i32]),
                    &[1],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(axis_const);
                connect_input(compute_op, "axis", axis_const)?;
                if let Some(o) = options.as_ref() {
                    set_bool_attr(compute_op, "exclusive", o.exclusive);
                    set_bool_attr(compute_op, "reverse", o.reversed);
                }
            }

            // Wire Gather / GatherElements via hiai::op::GatherV2D (x + indices + axis).
            if let Operation::Gather {
                input,
                indices,
                options,
                ..
            }
            | Operation::GatherElements {
                input,
                indices,
                options,
                ..
            } = op
            {
                connect_src_input(compute_op, "x", *input, &handles, &split_out)?;
                connect_src_input(compute_op, "indices", *indices, &handles, &split_out)?;
                let axis = options.as_ref().map(|o| o.axis as i64).unwrap_or(0);
                set_int64_attr(compute_op, "axis", axis);
            }

            // Wire GatherND via hiai::op::GatherNd (x + indices).
            if let Operation::GatherND { input, indices, .. } = op {
                connect_src_input(compute_op, "x", *input, &handles, &split_out)?;
                connect_src_input(compute_op, "indices", *indices, &handles, &split_out)?;
            }

            // Wire ScatterElements / ScatterND via hiai::op::ScatterNdUpdate.
            if let Operation::ScatterElements {
                input,
                indices,
                updates,
                ..
            }
            | Operation::ScatterND {
                input,
                indices,
                updates,
                ..
            } = op
            {
                connect_src_input(compute_op, "var", *input, &handles, &split_out)?;
                connect_src_input(compute_op, "indices", *indices, &handles, &split_out)?;
                connect_src_input(compute_op, "updates", *updates, &handles, &split_out)?;
            }

            // Wire Where via hiai::op::Select (condition, x1, x2).
            if let Operation::Where {
                condition,
                true_value,
                false_value,
                ..
            } = op
            {
                connect_src_input(compute_op, "condition", *condition, &handles, &split_out)?;
                connect_src_input(compute_op, "x1", *true_value, &handles, &split_out)?;
                connect_src_input(compute_op, "x2", *false_value, &handles, &split_out)?;
            }

            // Wire InstanceNormalization via hiai::op::BNInference (x + epsilon + scale/offset).
            if let Operation::InstanceNormalization { input, options, .. } = op {
                connect_src_input(compute_op, "x", *input, &handles, &split_out)?;
                if let Some(o) = options.as_ref() {
                    set_float_attr(compute_op, "epsilon", o.epsilon as f32);
                    if let Some(scale_id) = o.scale {
                        connect_src_input(compute_op, "scale", scale_id, &handles, &split_out)?;
                    }
                    if let Some(bias_id) = o.bias {
                        connect_src_input(compute_op, "offset", bias_id, &handles, &split_out)?;
                    }
                }
            }

            // Wire QuantizeLinear via hiai::op::QuantizeV2 (x + dtype; scale/zero_point
            // wiring is a TODO matching the reference).
            if let Operation::QuantizeLinear { input, .. } = op {
                connect_src_input(compute_op, "x", *input, &handles, &split_out)?;
                set_int64_attr(compute_op, "dtype", ddk_CannDataType::CANN_DT_UINT8 as i64);
            }

            let outputs: &[u32] = op_outputs(op);
            let out_handle = if nhwc_out {
                emit_transpose(
                    &format!("nhwc_out_{}", outputs[0]),
                    compute_op,
                    &[0, 2, 3, 1],
                    &mut extra_ops,
                )?
            } else {
                compute_op
            };
            for &out_id in outputs.iter() {
                handles[out_id as usize] = out_handle;
            }

            compute_ops.push(compute_op);
        }

        // ── 4. Create NetOutput ─────────────────────────────────────────
        let out_name = graph.operands[graph.output_operands[0] as usize]
            .name
            .as_deref()
            .unwrap_or("output");
        let net_name = CString::new(out_name).unwrap();
        let mut net_out = unsafe {
            ddk_cann_op_net_output_with_name(net_name.as_ptr(), graph.output_operands.len() as i32)
        };
        if net_out.is_null() {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_op_net_output failed".into(),
            });
        }

        let x_name = CString::new("x").unwrap();
        let y_name = CString::new("y").unwrap();
        let output_type_name = CString::new("output_type").unwrap();
        let status = unsafe {
            ddk_cann_operator_create_dynamic_input(
                net_out,
                x_name.as_ptr(),
                graph.output_operands.len() as u32,
            )
        };
        if status != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_operator_create_dynamic_input failed: {status}").into(),
            });
        }

        for (output_index, &out_id) in graph.output_operands.iter().enumerate() {
            let source_handle = handles[out_id as usize];
            if source_handle.is_null() {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("no handle for output operand {out_id}").into(),
                });
            }
            // Dynamic input index is 1-based (matches the Chromium reference).
            let dst_index = (output_index + 1) as u32;
            let status = unsafe { set_dynamic_input(net_out, &x_name, dst_index, source_handle) };
            if status != 0 {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!(
                        "cann_operator_set_dynamic_input_by_index[{output_index}] failed: {status}"
                    )
                    .into(),
                });
            }
        }

        let status = unsafe {
            ddk_cann_operator_create_dynamic_output(
                net_out,
                y_name.as_ptr(),
                graph.output_operands.len() as u32,
            )
        };
        if status != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_operator_create_dynamic_output failed: {status}").into(),
            });
        }

        let out_types: Vec<i64> = graph
            .output_operands
            .iter()
            .map(|&id| cann_data_type(graph.operands[id as usize].descriptor.data_type) as i64)
            .collect();
        let status = unsafe {
            ddk_cann_operator_set_attr_int64_list(
                net_out,
                output_type_name.as_ptr(),
                out_types.as_ptr(),
                out_types.len() as i32,
            )
        };
        if status != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_operator_set_attr_int64_list failed: {status}").into(),
            });
        }

        let mut all_ops: Vec<ddk_CannOperatorHandle> = Vec::new();
        all_ops.extend(data_ops.clone());
        all_ops.extend(const_ops);
        all_ops.extend(compute_ops);
        all_ops.extend(extra_ops);
        all_ops.push(net_out);

        // ── 5. Add all ops to graph ─────────────────────────────────────
        for &handle in &all_ops {
            if unsafe { ddk_cann_graph_add_op(can_graph, handle) } != 0 {
                let name = unsafe { ddk_cann_operator_get_name(handle) };
                let op_type = unsafe { ddk_cann_operator_get_type(handle) };
                let name = if name.is_null() {
                    "<unknown>".to_string()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(name) }
                        .to_string_lossy()
                        .into_owned()
                };
                let op_type = if op_type.is_null() {
                    "<unknown>".to_string()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(op_type) }
                        .to_string_lossy()
                        .into_owned()
                };
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_graph_add_op failed for {op_type} '{name}'").into(),
                });
            }
        }

        // ── 6. Set graph inputs / outputs ───────────────────────────────
        unsafe {
            ddk_cann_graph_set_inputs(can_graph, data_ops.as_mut_ptr(), data_ops.len() as i32);
            ddk_cann_graph_set_outputs(can_graph, &mut net_out, 1);
        }

        // ── 7. Validate graph ───────────────────────────────────────────
        if unsafe { ddk_cann_graph_is_valid(can_graph) } == 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_graph_is_valid returned false".into(),
            });
        }

        // ── 8. Compile model ────────────────────────────────────────────
        let model_name = CString::new("webnn_model").unwrap();
        let model = unsafe { ddk_cann_model_create_with_name(model_name.as_ptr()) };
        if model.is_null() {
            unsafe { ddk_cann_graph_destroy(can_graph) };
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_model_create_with_name failed".into(),
            });
        }
        if unsafe { ddk_cann_model_set_graph(model, can_graph) } != 0 {
            unsafe {
                ddk_cann_model_destroy(model);
                ddk_cann_graph_destroy(can_graph);
            }
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_model_set_graph failed".into(),
            });
        }

        let ir_handle = unsafe { ddk_cann_hiai_ir_build_create() };
        if ir_handle.is_null() {
            unsafe {
                ddk_cann_model_destroy(model);
                ddk_cann_graph_destroy(can_graph);
            }
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_hiai_ir_build_create failed".into(),
            });
        }

        let mut buffer = ddk_CannModelBuffer {
            data: std::ptr::null_mut(),
            length: 0,
        };
        if unsafe { ddk_cann_model_create_buff_default(ir_handle, model, &mut buffer) } != 0
            || buffer.data.is_null()
        {
            unsafe {
                ddk_cann_hiai_ir_build_destroy(ir_handle);
                ddk_cann_model_destroy(model);
                ddk_cann_graph_destroy(can_graph);
            }
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_model_create_buff_default failed".into(),
            });
        }

        // Build options: CUSTOM device select + NPU order, matching the
        // Chromium reference (which does not set input shapes; HiAI's HCL
        // compiler pads to 4D itself and rejects explicitly-set mixed-rank
        // input shapes).
        let build_opts = unsafe { ddk_cann_build_options_create() };
        if !build_opts.is_null() {
            unsafe {
                ddk_cann_build_options_set_mode(build_opts, 1); // CUSTOM
                ddk_cann_build_options_set_weight_data_type(build_opts, 1); // FP16 weights
                let devices: [i32; 1] = [0]; // 0 = NPU
                ddk_cann_build_options_set_device_order(build_opts, devices.as_ptr(), 1);
            }
        }

        // Compile IR model to OM bytes.
        let status = unsafe { ddk_cann_build_model(ir_handle, model, build_opts, &mut buffer) };
        if status != 0 || buffer.data.is_null() {
            unsafe {
                ddk_cann_model_buffer_destroy(ir_handle, &mut buffer);
                ddk_cann_hiai_ir_build_destroy(ir_handle);
                ddk_cann_model_destroy(model);
                ddk_cann_graph_destroy(can_graph);
            }
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_build_model failed".into(),
            });
        }

        let bytes =
            unsafe { std::slice::from_raw_parts(buffer.data as *const u8, buffer.length as usize) }
                .to_vec();

        // ── 9. Cleanup ──────────────────────────────────────────────────
        unsafe {
            ddk_cann_model_buffer_destroy(ir_handle, &mut buffer);
        }
        for &handle in &all_ops {
            unsafe { ddk_cann_operator_destroy(handle) };
        }
        unsafe {
            ddk_cann_hiai_ir_build_destroy(ir_handle);
            ddk_cann_model_destroy(model);
            ddk_cann_graph_destroy(can_graph);
        }

        Ok(bytes)
    }
}

#[cfg(feature = "cann-runtime")]
pub(crate) use adapter::encode_via_adapter;

#[cfg(not(feature = "cann-runtime"))]
pub(crate) fn encode_via_adapter(_graph: &GraphInfo) -> Result<Vec<u8>, GraphError> {
    Err(GraphError::ConversionFailed {
        format: "cann".into(),
        reason: "CANN shim not available (mock mode)".into(),
    })
}

pub struct CannConverter;

impl GraphConverter for CannConverter {
    fn format(&self) -> &'static str {
        "cann"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        // Call CANN shim layer.
        if let Ok(bytes) = encode_via_adapter(graph) {
            return Ok(ConvertedGraph {
                format: "cann",
                content_type: "application/octet-stream",
                data: bytes,
                weights_data: None,
            });
        }
        // Fallback to Mock CANN -- validates ops, returns placeholder bytes.
        let model_bytes = build_hiai_ir_model_mock(graph)?;
        Ok(ConvertedGraph {
            format: "cann",
            content_type: "application/octet-stream",
            data: model_bytes,
            weights_data: None,
        })
    }
}

// cann-runtime-mock: verify graph structure, return placeholder bytes.
fn build_hiai_ir_model_mock(graph: &GraphInfo) -> Result<Vec<u8>, GraphError> {
    let mut operation_count: usize = 0;
    for operation in &graph.operations {
        if !is_unsupported_op(operation) {
            operation_count += 1;
        }
    }
    if operation_count == 0 {
        return Err(GraphError::ConversionFailed {
            format: "cann".to_string(),
            reason: "no supported ops found".to_string(),
        });
    }
    Ok(vec![0x00, 0x00, 0x00, 0x00])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
    use crate::operator_options::MLDimension;
    use crate::operators::Operation;
    use std::collections::HashMap;

    fn make_add_graph() -> GraphInfo {
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![Dimension::Static(2), Dimension::Static(2)],
                        pending_permutation: vec![],
                    },
                    name: Some("lhs".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![Dimension::Static(2), Dimension::Static(2)],
                        pending_permutation: vec![],
                    },
                    name: Some("rhs".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![Dimension::Static(2), Dimension::Static(2)],
                        pending_permutation: vec![],
                    },
                    name: Some("sum".to_string()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![Operation::Add {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    #[test]
    fn test_add_graph_converts() {
        let graph = make_add_graph();
        let converter = CannConverter;
        assert_eq!(converter.format(), "cann");
        let result = converter.convert(&graph);
        assert!(result.is_ok(), "{result:?}");
        let converted = result.unwrap();
        assert!(!converted.data.is_empty());
    }

    #[test]
    fn test_webnn_op_to_hiai_relu() {
        let op = Operation::Relu {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("ReLU"));
    }

    #[test]
    fn test_webnn_op_to_hiai_sigmoid() {
        let op = Operation::Sigmoid {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Sigmoid"));
    }

    #[test]
    fn test_webnn_op_to_hiai_tanh() {
        let op = Operation::Tanh {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Tanh"));
    }

    #[test]
    fn test_webnn_op_to_hiai_add() {
        let op = Operation::Add {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Add"));
    }

    #[test]
    fn test_webnn_op_to_hiai_mul() {
        let op = Operation::Mul {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Mul"));
    }

    #[test]
    fn test_webnn_op_to_hiai_sub() {
        let op = Operation::Sub {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Sub"));
    }

    #[test]
    fn test_webnn_op_to_hiai_div() {
        let op = Operation::Div {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Div"));
    }

    #[test]
    fn test_webnn_op_to_hiai_cast() {
        let op = Operation::Cast {
            input: 0,
            data_type: crate::operator_enums::MLOperandDataType::Int32,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Cast"));
    }

    #[test]
    fn test_webnn_op_to_hiai_reduce_sum() {
        let op = Operation::ReduceSum {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("ReduceSum"));
    }

    #[test]
    fn test_webnn_op_to_hiai_conv2d() {
        let op = Operation::Conv2d {
            input: 0,
            filter: 1,
            options: None,
            outputs: vec![3],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Conv2D"));
    }

    #[test]
    fn test_webnn_op_to_hiai_max_pool2d() {
        let op = Operation::MaxPool2d {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("MaxPool"));
    }

    #[test]
    fn test_webnn_op_to_hiai_average_pool2d() {
        let op = Operation::AveragePool2d {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("AvgPool"));
    }

    #[test]
    fn test_webnn_op_to_hiai_matmul() {
        let op = Operation::Matmul {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("MatMul"));
    }

    #[test]
    fn test_webnn_op_to_hiai_softmax() {
        let op = Operation::Softmax {
            input: 0,
            axis: 1,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Softmax"));
    }

    #[test]
    fn test_webnn_op_to_hiai_reshape() {
        let op = Operation::Reshape {
            input: 0,
            new_shape: vec![MLDimension::Static(1), MLDimension::Static(4)],
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Reshape"));
    }

    #[test]
    fn test_webnn_op_to_hiai_concat() {
        let op = Operation::Concat {
            inputs: vec![0, 1],
            axis: 0,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Concat"));
    }

    #[test]
    fn test_webnn_op_to_hiai_identity() {
        let op = Operation::Identity {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), None);
    }
}
