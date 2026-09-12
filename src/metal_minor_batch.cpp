/*
Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Keep existing fused kernels, but own their intermediate
// buffers in a single primitive so GPU lifetime tracking cannot retain one
// allocation per half-step. No input or externally visible snapshot is reused.
#include "metal_minor_batch.h"

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/fast_primitives.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace mlxpdlp::detail {
namespace {

namespace batch_mx = mlx::core;
using BatchKernelState = decltype(std::declval<batch_mx::fast::CustomKernel>().state());

struct BatchKernel {
    BatchKernelState state;
    std::vector<int> input_map;
    int vector_index;
    int current_index;
};

int batch_add_input(const batch_mx::array &input, std::vector<batch_mx::array> &inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].id() == input.id())
            return static_cast<int>(i);
    }
    inputs.push_back(input);
    return static_cast<int>(inputs.size() - 1);
}

BatchKernel capture_batch_kernel(const batch_mx::array &output,
                                 std::vector<batch_mx::array> &inputs, bool primal) {
    const auto *primitive =
        dynamic_cast<batch_mx::fast::CustomKernel *>(output.primitive_ptr().get());
    if (!primitive)
        throw std::logic_error("batch requires a custom kernel");
    auto state = primitive->state();
    const auto &source_inputs = output.inputs();
    const bool adaptive = std::get<0>(state).find("_adaptive_") != std::string::npos;
    const int vector_index = static_cast<int>(source_inputs.size()) - (primal ? 7 : 6);
    if (std::get<0>(state).find(primal ? "custom_kernel_mlxpdlp_fused_primal_"
                                       : "custom_kernel_mlxpdlp_fused_dual_") != 0 ||
        std::get<0>(state).find("_minor") == std::string::npos || std::get<6>(state).has_value() ||
        !std::get<7>(state).empty() || std::get<8>(state) || std::get<9>(state) != 0 ||
        source_inputs.size() != static_cast<size_t>((primal ? 10 : 9) + (adaptive ? 2 : 0)) ||
        std::get<4>(state).size() != source_inputs.size()) {
        throw std::logic_error("unsupported batch kernel metadata");
    }
    std::vector<int> map(source_inputs.size(), -1);
    for (size_t i = 0; i + 1 < source_inputs.size(); ++i) {
        if (static_cast<int>(i) != vector_index && static_cast<int>(i) != vector_index + 1) {
            map[i] = batch_add_input(source_inputs[i], inputs);
        }
    }
    if (source_inputs.back().dtype() != batch_mx::float32 ||
        source_inputs.back().shape() != batch_mx::Shape{3} ||
        std::get<4>(state).back() != std::tuple{false, false, false}) {
        throw std::logic_error("batch scalar shape metadata is unsupported");
    }
    return {std::move(state), std::move(map), vector_index, vector_index + 1};
}

// The pinned MLX revision has ten metadata fields. Newer MLX adds serialized
// compile options as the eleventh field. Preserve those options in compilation
// and the cache key, rather than relying on a version-specific default.
template <class Device, class State>
MTL::Library *batch_library(Device &device, const State &state) {
    static_assert(std::tuple_size_v<State> == 10 || std::tuple_size_v<State> == 11);
    const auto &source = std::get<1>(state);
    std::string key = "mlxpdlp_batch_" + std::get<0>(state) + "_" +
                      std::to_string(std::hash<std::string>{}(source));
    auto builder = [&source] { return batch_mx::metal::utils() + source; };
    if constexpr (std::tuple_size_v<State> == 11) {
        const auto options = std::get<10>(state);
        key += "_" + std::to_string(options);
        return device.get_library(key, options, builder);
    } else {
        return device.get_library(key, builder);
    }
}

class MetalMinorBatch final : public batch_mx::Primitive {
  public:
    MetalMinorBatch(batch_mx::Stream stream, BatchKernel primal, BatchKernel dual, int x_input,
                    int y_input, std::vector<BatchScalars> scalars)
        : Primitive(stream), primal_(std::move(primal)), dual_(std::move(dual)), x_input_(x_input),
          y_input_(y_input), scalars_(std::move(scalars)) {}

    const char *name() const override {
        return "MlxPdlpMetalMinorBatch";
    }

    void eval_cpu(const std::vector<batch_mx::array> &, std::vector<batch_mx::array> &) override {
        throw std::runtime_error("Metal iteration batches require a GPU");
    }

    void eval_gpu(const std::vector<batch_mx::array> &inputs,
                  std::vector<batch_mx::array> &outputs) override {
        for (const auto &input : inputs) {
            if (!input.flags().row_contiguous) {
                throw std::runtime_error("batch requires contiguous inputs");
            }
        }
        for (auto &output : outputs) {
            output.set_data(batch_mx::allocator::malloc(output.nbytes()));
        }
        std::vector<batch_mx::array> scratch;
        if (scalars_.size() > 1) {
            for (int index : {0, 2}) {
                scratch.emplace_back(outputs[index].shape(), outputs[index].dtype(), nullptr,
                                     std::vector<batch_mx::array>{});
                scratch.back().set_data(batch_mx::allocator::malloc(scratch.back().nbytes()));
            }
        }
        auto &device = batch_mx::metal::device(stream().device);
        auto load = [&](const BatchKernel &plan) {
            const auto &name = std::get<0>(plan.state);
            auto library = batch_library(device, plan.state);
            auto *kernel = device.get_kernel(name, library);
            const auto [gx, gy, gz] = std::get<2>(plan.state);
            const auto [tx, ty, tz] = std::get<3>(plan.state);
            if (gx <= 0 || gy <= 0 || gz <= 0 || tx <= 0 || ty <= 0 || tz <= 0 ||
                static_cast<size_t>(tx) * ty * tz > kernel->maxTotalThreadsPerThreadgroup()) {
                throw std::runtime_error("invalid Metal batch launch dimensions");
            }
            return kernel;
        };
        auto *primal_kernel = load(primal_);
        auto *dual_kernel = load(dual_);
        auto &encoder = batch_mx::metal::get_command_encoder(stream());
        // Retain all owned buffers before the first dispatch, including when
        // a later encoding operation throws. They outlive GPU execution.
        auto retained = outputs;
        retained.insert(retained.end(), scratch.begin(), scratch.end());
        encoder.add_temporaries(std::move(retained));

        auto encode = [&](const BatchKernel &plan, MTL::ComputePipelineState *kernel,
                          const batch_mx::array &vector, const batch_mx::array &current,
                          const std::array<float, 3> &scalars, batch_mx::array &next,
                          batch_mx::array *reflected) {
            encoder.set_compute_pipeline_state(kernel);
            int binding = 0;
            const auto &shape_infos = std::get<4>(plan.state);
            for (size_t i = 0; i < plan.input_map.size(); ++i) {
                if (i + 1 == plan.input_map.size()) {
                    encoder.set_bytes(scalars.data(), 3, binding++);
                    continue;
                }
                const auto &input = static_cast<int>(i) == plan.vector_index ? vector
                                    : static_cast<int>(i) == plan.current_index
                                        ? current
                                        : inputs[plan.input_map[i]];
                encoder.set_input_array(input, binding++);
                if (input.ndim() > 0) {
                    const int ndim = input.ndim();
                    if (std::get<0>(shape_infos[i])) {
                        encoder.set_vector_bytes(input.shape(), ndim, binding++);
                    }
                    if (std::get<1>(shape_infos[i])) {
                        encoder.set_vector_bytes(input.strides(), ndim, binding++);
                    }
                    if (std::get<2>(shape_infos[i])) {
                        encoder.set_bytes(ndim, binding++);
                    }
                }
            }
            encoder.set_output_array(next, binding++);
            if (reflected)
                encoder.set_output_array(*reflected, binding++);
            const auto [gx, gy, gz] = std::get<2>(plan.state);
            const auto [tx, ty, tz] = std::get<3>(plan.state);
            encoder.dispatch_threads(
                MTL::Size(gx, gy, gz),
                MTL::Size(std::min(gx, tx), std::min(gy, ty), std::min(gz, tz)));
        };

        auto x = inputs[x_input_];
        auto y = inputs[y_input_];
        for (size_t i = 0; i < scalars_.size(); ++i) {
            // Last iteration always writes the public outputs. Reusing the
            // reflected output is safe: each dual dispatch consumes it before
            // the next primal dispatch. MLX's encoder inserts RAW/WAR barriers.
            const bool alternate = (scalars_.size() - 1 - i) % 2 != 0;
            auto &next_x = alternate ? scratch[0] : outputs[0];
            auto &next_y = alternate ? scratch[1] : outputs[2];
            encode(primal_, primal_kernel, y, x, scalars_[i][0], next_x, &outputs[1]);
            encode(dual_, dual_kernel, outputs[1], y, scalars_[i][1], next_y, nullptr);
            x = next_x;
            y = next_y;
        }
    }

  private:
    BatchKernel primal_;
    BatchKernel dual_;
    int x_input_;
    int y_input_;
    std::vector<BatchScalars> scalars_;
};

} // namespace

std::vector<batch_mx::array> metal_minor_batch(const std::vector<batch_mx::array> &primal,
                                               const std::vector<batch_mx::array> &dual,
                                               std::vector<BatchScalars> scalars,
                                               batch_mx::Stream stream) {
    if (scalars.empty() || scalars.size() > max_metal_batch_iterations || primal.size() != 2 ||
        dual.size() != 1) {
        throw std::logic_error("invalid Metal iteration batch");
    }
    for (const auto *outputs : {&primal, &dual}) {
        for (const auto &output : *outputs) {
            if (output.dtype() != batch_mx::float32 || output.ndim() != 1 || output.size() == 0 ||
                !output.has_primitive() || output.primitive().stream() != stream) {
                throw std::logic_error("invalid Metal batch prototype");
            }
        }
    }
    std::vector<batch_mx::array> inputs;
    auto primal_plan = capture_batch_kernel(primal[0], inputs, true);
    auto dual_plan = capture_batch_kernel(dual[0], inputs, false);
    const auto &x = primal[0].inputs()[primal_plan.current_index];
    const auto &y = primal[0].inputs()[primal_plan.vector_index];
    if (y.id() != dual[0].inputs()[dual_plan.current_index].id() ||
        x.shape() != primal[0].shape() || x.shape() != primal[1].shape() ||
        x.shape() != dual[0].inputs()[dual_plan.vector_index].shape() ||
        y.shape() != dual[0].shape()) {
        throw std::logic_error("batch initial states differ");
    }
    const int x_input = batch_add_input(x, inputs);
    const int y_input = batch_add_input(y, inputs);
    auto primitive = std::make_shared<MetalMinorBatch>(
        stream, std::move(primal_plan), std::move(dual_plan), x_input, y_input, std::move(scalars));
    return batch_mx::array::make_arrays({primal[0].shape(), primal[1].shape(), dual[0].shape()},
                                        {primal[0].dtype(), primal[1].dtype(), dual[0].dtype()},
                                        std::move(primitive), inputs);
}


namespace {
BatchKernel capture_spmm_batch_kernel(const batch_mx::array &output,
                                      std::vector<batch_mx::array> &inputs, bool primal) {
    const auto *primitive = dynamic_cast<batch_mx::fast::CustomKernel *>(output.primitive_ptr().get());
    if (!primitive) throw std::logic_error("SpMM batch requires a custom kernel");
    auto state = primitive->state();
    const auto &source = output.inputs();
    const auto &name = std::get<0>(state);
    const std::string suffix = primal ? "_1_minor" : "_2_minor";
    if (name.find("custom_kernel_mlxpdlp_spmm_") != 0 ||
        name.find(suffix + "_") == std::string::npos ||
        source.size() != 12 || std::get<4>(state).size() != source.size() ||
        std::get<6>(state).has_value() || !std::get<7>(state).empty() || std::get<8>(state) || std::get<9>(state) != 0)
        throw std::logic_error("unsupported SpMM batch metadata: " + name +
                               " (inputs=" + std::to_string(source.size()) + ")");
    if (source[11].dtype() != batch_mx::float32 || source[11].ndim() != 2 || source[11].shape(0) != 4 ||
        source[11].shape(1) != output.shape(1) || std::get<4>(state)[11] != std::tuple{false,false,false})
        throw std::logic_error("unsupported SpMM coefficient metadata");
    std::vector<int> map(12,-1);
    for (int i=0;i<11;++i) if (i!=3 && i!=5) map[i]=batch_add_input(source[i],inputs);
    return {std::move(state),std::move(map),3,5};
}

class MetalSpmmMinorBatch final : public batch_mx::Primitive {
  public:
    MetalSpmmMinorBatch(batch_mx::Stream stream, BatchKernel primal, BatchKernel dual,
                       int x, int y, std::vector<std::vector<float>> coefficients)
        : Primitive(stream), primal_(std::move(primal)),dual_(std::move(dual)),x_(x),y_(y),coefficients_(std::move(coefficients)) {}
    const char *name() const override { return "MlxPdlpMetalSpmmMinorBatch"; }
    void eval_cpu(const std::vector<batch_mx::array> &, std::vector<batch_mx::array> &) override {
        throw std::runtime_error("SpMM iteration batches require Metal");
    }
    void eval_gpu(const std::vector<batch_mx::array> &inputs, std::vector<batch_mx::array> &outputs) override {
        for (const auto &input : inputs) if (!input.flags().row_contiguous)
            throw std::runtime_error("SpMM batch requires contiguous inputs");
        for (auto &output : outputs) output.set_data(batch_mx::allocator::malloc(output.nbytes()));
        std::vector<batch_mx::array> scratch;
        if (coefficients_.size()>1) for (int index : {0,2}) {
            scratch.emplace_back(outputs[index].shape(),outputs[index].dtype(),nullptr,std::vector<batch_mx::array>{});
            scratch.back().set_data(batch_mx::allocator::malloc(scratch.back().nbytes()));
        }
        auto &device=batch_mx::metal::device(stream().device);
        auto load=[&](const BatchKernel &plan) {
            auto *kernel=device.get_kernel(std::get<0>(plan.state),batch_library(device,plan.state));
            const auto [gx,gy,gz]=std::get<2>(plan.state);
            const auto [tx,ty,tz]=std::get<3>(plan.state);
            if (gx<=0||gy<=0||gz<=0||tx<=0||ty<=0||tz<=0||size_t(tx)*ty*tz>kernel->maxTotalThreadsPerThreadgroup())
                throw std::runtime_error("invalid SpMM batch launch dimensions");
            return kernel;
        };
        auto *primal_kernel=load(primal_), *dual_kernel=load(dual_);
        auto &encoder=batch_mx::metal::get_command_encoder(stream());
        auto retained=outputs;retained.insert(retained.end(),scratch.begin(),scratch.end());
        encoder.add_temporaries(std::move(retained));
        auto encode=[&](const BatchKernel &plan,MTL::ComputePipelineState *kernel,
                        const batch_mx::array &vector,const batch_mx::array &current,
                        const std::vector<float> &coeff,batch_mx::array &next,batch_mx::array &reflected) {
            encoder.set_compute_pipeline_state(kernel);
            int binding=0;
            const auto &info=std::get<4>(plan.state);
            for (int i=0;i<12;++i) {
                if (i==11) {encoder.set_bytes(coeff.data(),coeff.size(),binding++);continue;}
                const auto &input=i==3?vector:i==5?current:inputs[plan.input_map[i]];
                encoder.set_input_array(input,binding++);
                const int ndim=input.ndim();
                if (ndim>0) {
                    if (std::get<0>(info[i])) encoder.set_vector_bytes(input.shape(),ndim,binding++);
                    if (std::get<1>(info[i])) encoder.set_vector_bytes(input.strides(),ndim,binding++);
                    if (std::get<2>(info[i])) encoder.set_bytes(ndim,binding++);
                }
            }
            encoder.set_output_array(next,binding++);encoder.set_output_array(reflected,binding++);
            const auto [gx,gy,gz]=std::get<2>(plan.state);const auto [tx,ty,tz]=std::get<3>(plan.state);
            encoder.dispatch_threads(MTL::Size(gx,gy,gz),MTL::Size(std::min(gx,tx),std::min(gy,ty),std::min(gz,tz)));
        };
        auto x=inputs[x_],y=inputs[y_];
        for (size_t i=0;i<coefficients_.size();++i) {
            const bool alternate=(coefficients_.size()-1-i)%2!=0;
            auto &next_x=alternate?scratch[0]:outputs[0];auto &next_y=alternate?scratch[1]:outputs[2];
            encode(primal_,primal_kernel,y,x,coefficients_[i],next_x,outputs[1]);
            encode(dual_,dual_kernel,outputs[1],y,coefficients_[i],next_y,outputs[3]);
            x=next_x;y=next_y;
        }
    }
  private:
    BatchKernel primal_,dual_;int x_,y_;std::vector<std::vector<float>> coefficients_;
};
} // namespace

std::vector<batch_mx::array> metal_spmm_minor_batch(const std::vector<batch_mx::array> &primal,
    const std::vector<batch_mx::array> &dual,std::vector<std::vector<float>> coefficients,batch_mx::Stream stream) {
    if (primal.size()!=2||dual.size()!=2||coefficients.empty()||coefficients.size()>max_metal_batch_iterations)
        throw std::logic_error("invalid SpMM iteration batch");
    for (const auto *group : {&primal,&dual}) for (const auto &a : *group)
        if (a.dtype()!=batch_mx::float32||a.ndim()!=2||a.size()==0||a.shape(1)>256||
            !a.has_primitive()||a.primitive().stream()!=stream)
            throw std::logic_error("invalid SpMM batch prototype");
    for (const auto &c : coefficients) if (c.size()!=size_t(4*primal[0].shape(1)))
        throw std::logic_error("invalid SpMM batch coefficients");
    std::vector<batch_mx::array> inputs;
    auto p=capture_spmm_batch_kernel(primal[0],inputs,true),d=capture_spmm_batch_kernel(dual[0],inputs,false);
    const auto &x=primal[0].inputs()[5];const auto &y=primal[0].inputs()[3];
    if (y.id()!=dual[0].inputs()[5].id()||x.shape()!=primal[0].shape()||x.shape()!=primal[1].shape()||
        y.shape()!=dual[0].shape()||y.shape()!=dual[1].shape()||primal[1].id()!=dual[0].inputs()[3].id())
        throw std::logic_error("SpMM prototype states differ");
    const int xi=batch_add_input(x,inputs),yi=batch_add_input(y,inputs);
    auto primitive=std::make_shared<MetalSpmmMinorBatch>(stream,std::move(p),std::move(d),xi,yi,std::move(coefficients));
    return batch_mx::array::make_arrays({x.shape(),x.shape(),y.shape(),y.shape()},
        {x.dtype(),x.dtype(),y.dtype(),y.dtype()},std::move(primitive),inputs);
}

} // namespace mlxpdlp::detail
