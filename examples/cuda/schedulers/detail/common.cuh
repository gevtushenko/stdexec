/*
 * Copyright (c) NVIDIA
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <execution.hpp>
#include <type_traits>

namespace example::cuda {

enum class device_type {
  host,
  device
};

#ifdef _NVHPC_CUDA
#include <nv/target>

__host__ __device__ inline device_type get_device_type() {
  if target (nv::target::is_host) {
    return device_type::host;
  }
  else {
    return device_type::device;
  }
}
#elif defined(__clang__) && defined(__CUDA__)
__host__ inline device_type get_device_type() { return device_type::host; }
__device__ inline device_type get_device_type() { return device_type::device; }
#endif

inline __host__ __device__ bool is_on_gpu() {
  return get_device_type() == device_type::device;
}

}

namespace example::cuda::stream {

  struct scheduler_t;

  struct receiver_base_t {};

  struct operation_state_base_t {
    bool owner_{false};
    cudaStream_t stream_{0};

    cudaStream_t allocate() {
      if (stream_ == 0) {
        owner_ = true;
        cudaStreamCreate(&stream_);
      }

      return stream_;
    }

    ~operation_state_base_t() {
      if (owner_) {
        cudaStreamDestroy(stream_);
        stream_ = 0;
      }
    }
  };

  namespace detail {
    template <class SenderId, class ReceiverId>
    struct operation_state_t : operation_state_base_t {
      using sender_t = std::__t<SenderId>;
      using receiver_t = std::__t<ReceiverId>;
      using inner_op_state_t = std::execution::connect_result_t<sender_t, receiver_t>;

      inner_op_state_t inner_op_;

      friend void tag_invoke(std::execution::start_t, operation_state_t& op) noexcept {
        op.stream_ = op.get_stream();
        std::execution::start(op.inner_op_);
      }

      cudaStream_t get_stream() {
        cudaStream_t stream{};

        if constexpr (std::is_base_of_v<operation_state_base_t, inner_op_state_t>) {
          stream = inner_op_.get_stream();
        } else {
          stream = allocate();
        }

        return stream;
      }

      template <class ReceiverProvider>
      operation_state_t(sender_t&& sender, ReceiverProvider receiver_provider)
        : inner_op_{std::execution::connect((sender_t&&)sender, receiver_provider(*this))}
      {}
    };
  }

  template <class Sender, class Receiver>
    using stream_op_state_t = detail::operation_state_t<
    std::__x<Sender>, 
    std::__x<Receiver>>;

  template <class Sender, class ReceiverProvider>
    stream_op_state_t<Sender, std::invoke_result_t<ReceiverProvider, operation_state_base_t&>> 
    stream_op_state(Sender&& sndr, ReceiverProvider receiver_provider) {
      return stream_op_state_t<Sender, std::invoke_result_t<ReceiverProvider, operation_state_base_t&>>(
          (Sender&&)sndr, receiver_provider);
    }

  template <class S>
  concept stream_sender = 
      std::execution::sender<S> &&
      std::is_same_v<
          std::tag_invoke_result_t<
            std::execution::get_completion_scheduler_t<std::execution::set_value_t>, S>,
          scheduler_t>;
}

