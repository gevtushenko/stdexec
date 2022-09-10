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

#include <thrust/device_vector.h>

#include "common.cuh"

namespace example::cuda::stream {

namespace then {

template <class Fun, class... As>
  __launch_bounds__(1) 
  __global__ void kernel(Fun fn, As... as) {
    fn(as...);
  }

template <class Fun, class ResultT, class... As>
  __launch_bounds__(1) 
  __global__ void kernel_with_result(Fun fn, ResultT* result, As... as) {
    *result = fn(as...);
  }

template <class ReceiverId, class Fun>
  class receiver_t
    : std::execution::receiver_adaptor<receiver_t<ReceiverId, Fun>, std::__t<ReceiverId>>
    , receiver_base_t {
    using Receiver = std::__t<ReceiverId>;
    friend std::execution::receiver_adaptor<receiver_t, Receiver>;

    Fun f_;
    operation_state_base_t &op_state_;

    template <class... As>
    void set_value(As&&... as) && noexcept 
      requires std::__callable<Fun, std::decay_t<As>...> {
      using result_t = std::decay_t<std::invoke_result_t<Fun, std::decay_t<As>...>>;

      if constexpr (std::is_same_v<void, result_t>) {
        kernel<Fun, std::decay_t<As>...><<<1, 1, 0, op_state_.stream_>>>(f_, as...);

        if constexpr (!std::is_base_of_v<receiver_base_t, Receiver>) {
          cudaStreamSynchronize(op_state_.stream_);
        }

        std::execution::set_value(std::move(this->base()));
      } else {
        result_t *d_result{};
        cudaStream_t stream = op_state_.stream_;
        cudaMallocAsync(&d_result, sizeof(result_t), stream);
        kernel_with_result<Fun, std::decay_t<As>...><<<1, 1, 0, stream>>>(f_, d_result, as...);

        result_t h_result;
        cudaMemcpy(&h_result, d_result, sizeof(result_t), cudaMemcpyDeviceToHost);
        std::execution::set_value(std::move(this->base()), h_result);
      }
    }

   public:
    explicit receiver_t(Receiver rcvr, Fun fun, operation_state_base_t &op_state)
      : std::execution::receiver_adaptor<receiver_t, Receiver>((Receiver&&) rcvr)
      , f_((Fun&&) fun)
      , op_state_(op_state)
    {}
  };

}

template <class SenderId, class FunId>
  struct then_sender_t {
    using Sender = std::__t<SenderId>;
    using Fun = std::__t<FunId>;

    Sender sndr_;
    Fun fun_;

    using set_error_t = 
      std::execution::completion_signatures<
        std::execution::set_error_t(std::exception_ptr)>;

    template <class Receiver>
      using receiver_t = then::receiver_t<std::__x<Receiver>, Fun>;

    template <class Self, class Env>
      using completion_signatures =
        std::execution::__make_completion_signatures<
          std::__member_t<Self, Sender>,
          Env,
          std::execution::__with_error_invoke_t<
            std::execution::set_value_t, 
            Fun, 
            std::__member_t<Self, Sender>, 
            Env>,
          std::__mbind_front_q<std::execution::__set_value_invoke_t, Fun>>;

    template <std::__decays_to<then_sender_t> Self, std::execution::receiver Receiver>
      requires std::execution::receiver_of<Receiver, completion_signatures<Self, std::execution::env_of_t<Receiver>>>
    friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
      -> stream_op_state_t<std::__member_t<Self, Sender>, receiver_t<Receiver>> {
        return stream_op_state<std::__member_t<Self, Sender>>(((Self&&)self).sndr_, [&](operation_state_base_t& stream_provider) -> receiver_t<Receiver> {
          return receiver_t<Receiver>((Receiver&&)rcvr, self.fun_, stream_provider);
        });
    }

    template <std::__decays_to<then_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> std::execution::dependent_completion_signatures<Env>;

    template <std::__decays_to<then_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> completion_signatures<Self, Env> requires true;

    template <std::execution::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
      requires std::__callable<Tag, const Sender&, As...>
    friend auto tag_invoke(Tag tag, const then_sender_t& self, As&&... as)
      noexcept(std::__nothrow_callable<Tag, const Sender&, As...>)
      -> std::__call_result_if_t<std::execution::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
      return ((Tag&&) tag)(self.sndr_, (As&&) as...);
    }
  };

}
