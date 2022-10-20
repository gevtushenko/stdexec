/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#include <stdexec/execution.hpp>
#include <type_traits>

#include "nvexec/stream/common.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {

namespace upon_stopped {

template <class Fun>
  __launch_bounds__(1)
  __global__ void kernel(Fun fn) {
    fn();
  }

template <class Fun, class ResultT>
  __launch_bounds__(1)
  __global__ void kernel_with_result(Fun fn, ResultT* result) {
    new (result) ResultT(fn());
  }

template <class T>
  inline constexpr std::size_t size_of_ = sizeof(T);

template <>
  inline constexpr std::size_t size_of_<void> = 0;

template <class ReceiverId, class Fun>
  class receiver_t : public stream_receiver_base {
    using result_t = std::decay_t<std::invoke_result_t<Fun>>;

    Fun f_;
    operation_state_base_t<ReceiverId> &op_state_;

  public:
    constexpr static std::size_t memory_allocation_size = size_of_<result_t>;

    friend void tag_invoke(std::execution::set_stopped_t, receiver_t&& self) noexcept {
      constexpr bool does_not_return_a_value = std::is_same_v<void, result_t>;
      cudaStream_t stream = self.op_state_.stream_;

      if constexpr (does_not_return_a_value) {
        kernel<Fun><<<1, 1, 0, stream>>>(self.f_);
        if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError()); status == cudaSuccess) {
          self.op_state_.propagate_completion_signal(std::execution::set_value);
        } else {
          self.op_state_.propagate_completion_signal(std::execution::set_error, std::move(status));
        }
      } else {
        result_t *d_result = reinterpret_cast<result_t*>(self.op_state_.temp_storage_);
        kernel_with_result<Fun><<<1, 1, 0, stream>>>(self.f_, d_result);
        if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError()); status == cudaSuccess) {
          self.op_state_.propagate_completion_signal(std::execution::set_value, *d_result);
        } else {
          self.op_state_.propagate_completion_signal(std::execution::set_error, std::move(status));
        }
      }
    }

    template <stdexec::__one_of<std::execution::set_value_t,
                               std::execution::set_error_t> Tag, 
              class... As _NVCXX_CAPTURE_PACK(As)>
      friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
        _NVCXX_EXPAND_PACK(As, as,
          self.op_state_.propagate_completion_signal(tag, (As&&)as...);
        );
      }

    friend std::execution::env_of_t<stdexec::__t<ReceiverId>>
    tag_invoke(std::execution::get_env_t, const receiver_t& self) {
      return std::execution::get_env(self.op_state_.receiver_);
    }

    explicit receiver_t(Fun fun, operation_state_base_t<ReceiverId> &op_state)
      : f_((Fun&&) fun)
      , op_state_(op_state)
    {}
  };

}

template <class SenderId, class FunId>
  struct upon_stopped_sender_t : stream_sender_base {
    using Sender = stdexec::__t<SenderId>;
    using Fun = stdexec::__t<FunId>;

    Sender sndr_;
    Fun fun_;

    using set_error_t =
      std::execution::completion_signatures<
        std::execution::set_error_t(std::exception_ptr)>;

    template <class Receiver>
      using receiver_t = upon_stopped::receiver_t<stdexec::__x<Receiver>, Fun>;

    template <class Self, class Env>
      using completion_signatures =
        stdexec::__make_completion_signatures<
          stdexec::__member_t<Self, Sender>,
          Env,
          stdexec::__with_error_invoke_t<
            std::execution::set_stopped_t,
            Fun,
            stdexec::__member_t<Self, Sender>,
            Env>,
          stdexec::__q<stdexec::__compl_sigs::__default_set_value>,
          stdexec::__q1<stdexec::__compl_sigs::__default_set_error>,
          stdexec::__set_value_invoke_t<Fun>>;

    template <stdexec::__decays_to<upon_stopped_sender_t> Self, std::execution::receiver Receiver>
      requires std::execution::receiver_of<Receiver, completion_signatures<Self, std::execution::env_of_t<Receiver>>>
    friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
      -> stream_op_state_t<stdexec::__member_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<stdexec::__member_t<Self, Sender>>(
            ((Self&&)self).sndr_,
            (Receiver&&)rcvr,
            [&](operation_state_base_t<stdexec::__x<Receiver>>& stream_provider) -> receiver_t<Receiver> {
              return receiver_t<Receiver>(self.fun_, stream_provider);
            });
    }

    template <stdexec::__decays_to<upon_stopped_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> std::execution::dependent_completion_signatures<Env>;

    template <stdexec::__decays_to<upon_stopped_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> completion_signatures<Self, Env> requires true;

    template <stdexec::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
      requires stdexec::__callable<Tag, const Sender&, As...>
    friend auto tag_invoke(Tag tag, const upon_stopped_sender_t& self, As&&... as)
      noexcept(stdexec::__nothrow_callable<Tag, const Sender&, As...>)
      -> stdexec::__call_result_if_t<stdexec::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
      return ((Tag&&) tag)(self.sndr_, (As&&) as...);
    }
  };

}
