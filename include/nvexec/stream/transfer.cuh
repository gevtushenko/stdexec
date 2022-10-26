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

#include "../../stdexec/execution.hpp"
#include <type_traits>

#include "common.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {
 
namespace transfer {
  template <class SenderId, class ReceiverId>
    struct operation_state_t : operation_state_base_t<ReceiverId> {
      using Sender = stdexec::__t<SenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      using Env = typename operation_state_base_t<ReceiverId>::env_t;
      using variant_t = variant_storage_t<Sender, Env>;

      struct receiver_t {
        operation_state_t &op_state_;

        template <stdexec::__one_of<std::execution::set_value_t,
                                    std::execution::set_error_t,
                                    std::execution::set_stopped_t> Tag,
                  class... As >
        friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
          Tag{}(std::move(self.op_state_.receiver_), (As&&)as...);
        }

        friend Env
        tag_invoke(std::execution::get_env_t, const receiver_t& self) {
          return self.operation_state_.make_env();
        }
      };

      using task_t = continuation_task_t<receiver_t, variant_t>;

      cudaError_t status_{cudaSuccess};
      context_state_t context_state_;

      variant_t *storage_;
      task_t *task_;

      ::cuda::std::atomic_flag started_;

      using enqueue_receiver = stream_enqueue_receiver<stdexec::__x<Env>, stdexec::__x<variant_t>>;
      using inner_op_state_t = std::execution::connect_result_t<Sender, enqueue_receiver>;
      inner_op_state_t inner_op_;

      friend void tag_invoke(std::execution::start_t, operation_state_t& op) noexcept {
        op.started_.test_and_set(::cuda::std::memory_order::relaxed);

        if (op.status_ != cudaSuccess) {
          // Couldn't allocate memory for operation state, complete with error
          std::execution::set_error(std::move(op.receiver_), std::move(op.status_));
          return;
        }

        std::execution::start(op.inner_op_);
      }

      operation_state_t(Sender&& sender, Receiver &&receiver, context_state_t context_state)
        : operation_state_base_t<ReceiverId>((Receiver&&)receiver, context_state, true)
        , context_state_(context_state)
        , storage_(queue::make_host<variant_t>(this->status_).release())
        , task_(queue::make_host<task_t>(this->status_, receiver_t{*this}, storage_, this->get_stream()).release())
        , started_(ATOMIC_FLAG_INIT)
        , inner_op_{
            std::execution::connect(
                (Sender&&)sender,
                enqueue_receiver{
                  this->make_env(), 
                  storage_, 
                  task_, 
                  context_state_.hub_->producer()})} {
        if (this->status_ == cudaSuccess) {
          this->status_ = task_->status_;
        }
      }

      STDEXEC_IMMOVABLE(operation_state_t);
    };
}

template <class SenderId>
  struct transfer_sender_t : stream_sender_base {
    using Sender = stdexec::__t<SenderId>;

    template <class Self, class Receiver>
      using op_state_th = 
        transfer::operation_state_t<
          stdexec::__x<stdexec::__member_t<Self, Sender>>, 
          stdexec::__x<Receiver>>;

    context_state_t context_state_;
    Sender sndr_;

    template <stdexec::__decays_to<transfer_sender_t> Self, std::execution::receiver Receiver>
      requires std::execution::sender_to<stdexec::__member_t<Self, Sender>, Receiver>
    friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
      -> op_state_th<Self, Receiver> {
      return op_state_th<Self, Receiver>{
        (Sender&&)self.sndr_, 
        (Receiver&&)rcvr,
        self.context_state_};
    }

    template <stdexec::__decays_to<transfer_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> std::execution::dependent_completion_signatures<Env>;

    template <stdexec::__decays_to<transfer_sender_t> _Self, class _Env>
      friend auto tag_invoke(std::execution::get_completion_signatures_t, _Self&&, _Env) ->
        std::execution::make_completion_signatures<
          stdexec::__member_t<_Self, Sender>,
          _Env,
          std::execution::completion_signatures<
            std::execution::set_stopped_t(),
            std::execution::set_error_t(cudaError_t)
          >
        > requires true;

    template <stdexec::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
      requires stdexec::__callable<Tag, const Sender&, As...>
    friend auto tag_invoke(Tag tag, const transfer_sender_t& self, As&&... as)
      noexcept(stdexec::__nothrow_callable<Tag, const Sender&, As...>)
      -> stdexec::__call_result_if_t<stdexec::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
      return ((Tag&&) tag)(self.sndr_, (As&&) as...);
    }

    transfer_sender_t(context_state_t context_state, Sender sndr)
      : context_state_(context_state)
      , sndr_{(Sender&&)sndr} {
    }
  };

}
