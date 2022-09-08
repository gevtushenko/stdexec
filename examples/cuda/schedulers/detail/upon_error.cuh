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

namespace upon_error {

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

    template <class Error>
    void set_error(Error&& error) && noexcept 
      requires std::__callable<Fun, Error> {
      using result_t = std::decay_t<std::invoke_result_t<Fun, std::decay_t<Error>>>;

      if constexpr (std::is_same_v<void, result_t>) {
        kernel<Fun, std::decay_t<Error>><<<1, 1, 0, op_state_.stream_>>>(f_, error);

        if constexpr (!std::is_base_of_v<receiver_base_t, Receiver>) {
          cudaStreamSynchronize(op_state_.stream_);
        }

        std::execution::set_value(std::move(this->base()));
      } else {
        cudaStream_t stream = op_state_.stream_;
        // TODO Replace device vector with simple allocation (memcpy async?) 
        // now thrust will not use the stream
        thrust::device_vector<result_t> result(1);
        kernel_with_result<Fun, std::decay_t<Error>><<<1, 1, 0, stream>>>(f_, thrust::raw_pointer_cast(result.data()), error);
        cudaStreamSynchronize(stream);
        std::execution::set_value(std::move(this->base()), static_cast<result_t>(result[0]));
      }
    }

   public:
    explicit receiver_t(Receiver rcvr, Fun fun, operation_state_base_t &op_state)
      : std::execution::receiver_adaptor<receiver_t, Receiver>((Receiver&&) rcvr)
      , f_((Fun&&) fun)
      , op_state_(op_state)
    {}
  };

template <class SenderId, class ReceiverId, class Fun>
  struct op_state_t : operation_state_base_t {
    using Sender = std::__t<SenderId>;
    using Receiver = std::__t<ReceiverId>;
    using upon_error_receiver_t = receiver_t<ReceiverId, Fun>;
    using inner_op_state = std::execution::connect_result_t<Sender, upon_error_receiver_t>;

    inner_op_state inner_op_;

    friend void tag_invoke(std::execution::start_t, op_state_t& op) noexcept {
      cudaStreamCreate(&op.stream_);
      std::execution::start(op.inner_op_);
    }

    operation_state_base_t& get_stream_provider() {
      if constexpr (std::is_base_of_v<operation_state_base_t, inner_op_state>) {
        return inner_op_.get_stream_provider();
      }

      return *this;
    }

    op_state_t(Fun fn, Sender&& sender, Receiver receiver)
      : inner_op_{ 
          std::execution::connect(
              (Sender&&)sender, 
              upon_error_receiver_t((Receiver&&)receiver, fn, get_stream_provider())) } 
    { }

    ~op_state_t() {
      if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = 0;
      }
    }
  };

}

template <class SenderId, class FunId>
  struct upon_error_sender_t {
    using Sender = std::__t<SenderId>;
    using Fun = std::__t<FunId>;

    Sender sndr_;
    Fun fun_;

    using set_error_t = 
      std::execution::completion_signatures<
        std::execution::set_error_t(std::exception_ptr)>;

    template <class Self, class Receiver>
      using operation_state_t = 
        upon_error::op_state_t<
          std::__x<std::__member_t<Self, Sender>>, 
          std::__x<std::remove_cvref_t<Receiver>>, 
          Fun>;

    template <class Self, class Env>
      using completion_signatures =
        std::execution::__make_completion_signatures<
          std::__member_t<Self, Sender>,
          Env,
          std::execution::__with_error_invoke_t<
            std::execution::set_error_t, 
            Fun, 
            std::__member_t<Self, Sender>, 
            Env>,
          std::__q<std::execution::__completion_signatures::__default_set_value>,
          std::__mbind_front_q<std::execution::__set_value_invoke_t, Fun>>;

    template <std::__decays_to<upon_error_sender_t> Self, std::execution::receiver Receiver>
      requires std::execution::receiver_of<Receiver, completion_signatures<Self, std::execution::env_of_t<Receiver>>>
    friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
      noexcept(std::is_nothrow_constructible_v<operation_state_t<Self, Receiver>, Fun, Sender, Receiver>) 
      -> operation_state_t<Self, Receiver> {
      return operation_state_t<Self, Receiver>(self.fun_, ((Self&&)self).sndr_, (Receiver&&)rcvr);
    }

    template <std::__decays_to<upon_error_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> std::execution::dependent_completion_signatures<Env>;

    template <std::__decays_to<upon_error_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> completion_signatures<Self, Env> requires true;

    template <std::execution::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
      requires std::__callable<Tag, const Sender&, As...>
    friend auto tag_invoke(Tag tag, const upon_error_sender_t& self, As&&... as)
      noexcept(std::__nothrow_callable<Tag, const Sender&, As...>)
      -> std::__call_result_if_t<std::execution::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
      return ((Tag&&) tag)(self.sndr_, (As&&) as...);
    }
  };

}
