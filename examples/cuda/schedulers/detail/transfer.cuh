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

#include "common.cuh"

namespace example::cuda::stream {

namespace transfer {

  template <std::size_t I, class T>
    void fetch(cudaStream_t stream, T&) {
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    }

  template <std::size_t I, class T, class Head, class... As>
    void fetch(cudaStream_t stream, T& tpl, Head&& head, As&&... as) {
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&std::get<I>(tpl),
                                          &head,
                                          sizeof(std::decay_t<Head>),
                                          cudaMemcpyDeviceToHost,
                                          stream));
      fetch<I + 1>(stream, tpl, (As&&)as...);
    }

  template <class ReceiverId>
  struct sink_receiver_t : receiver_base_t {
    using Receiver = std::__t<ReceiverId>;
    Receiver receiver_;

    template <class Tag, class... As _NVCXX_CAPTURE_PACK(As)>
      friend void tag_invoke(Tag, sink_receiver_t&& __rcvr, As&&... as) noexcept {
        _NVCXX_EXPAND_PACK(As, as,

        );
      }

    friend std::execution::env_of_t<Receiver> tag_invoke(std::execution::get_env_t, const sink_receiver_t& self) noexcept {
      return std::execution::get_env(self.receiver_);
    }
  };

  template <class SenderId, class ReceiverId>
    struct bypass_receiver_t : receiver_base_t {
      using Sender = std::__t<SenderId>;
      operation_state_base_t<ReceiverId>& operation_state_;

      template <std::__one_of<std::execution::set_value_t,
                              std::execution::set_error_t,
                              std::execution::set_stopped_t> Tag,
                class... As _NVCXX_CAPTURE_PACK(As)>
      friend void tag_invoke(Tag tag, bypass_receiver_t&& self, As&&... as) noexcept {
        auto stream = self.operation_state_.stream_;

        _NVCXX_EXPAND_PACK(As, as,
          if constexpr (gpu_stream_sender<Sender>) {
            std::tuple<std::decay_t<As>...> h_as;
            fetch<0>(stream, h_as, (As&&)as...);

            std::apply([&](auto&&... tas) {
              tag(std::move(self.operation_state_.receiver_.receiver_), tas...);
            }, h_as);
          } else {
            THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
            tag(std::move(self.operation_state_.receiver_.receiver_), (As&&)as...);
          }
        );
      }

      friend std::execution::env_of_t<typename std::__t<ReceiverId>::Receiver>
      tag_invoke(std::execution::get_env_t, const bypass_receiver_t& self) {
        return std::execution::get_env(self.operation_state_.receiver_.receiver_);
      }
    };
}

template <class SenderId>
  struct transfer_sender_t : sender_base_t {
    using Sender = std::__t<SenderId>;

    detail::queue::task_hub_t* hub_;
    Sender sndr_;

    template <class Receiver>
      using receiver_t = transfer::bypass_receiver_t<SenderId, std::__x<transfer::sink_receiver_t<std::__x<Receiver>>>>;

    template <std::__decays_to<transfer_sender_t> Self, std::execution::receiver Receiver>
      requires std::execution::sender_to<std::__member_t<Self, Sender>, Receiver>
    friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
      -> stream_op_state_t<std::__member_t<Self, Sender>, receiver_t<Receiver>, transfer::sink_receiver_t<std::__x<Receiver>>> {
        return stream_op_state<std::__member_t<Self, Sender>>(
          self.hub_,
          ((Self&&)self).sndr_,
          transfer::sink_receiver_t<std::__x<Receiver>>{{}, (Receiver&&)rcvr},
          [&](operation_state_base_t<std::__x<transfer::sink_receiver_t<std::__x<Receiver>>>>& stream_provider) -> receiver_t<Receiver> {
            return receiver_t<Receiver>{{}, stream_provider};
          });
    }

    template <std::__decays_to<transfer_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> std::execution::dependent_completion_signatures<Env>;

    template <std::__decays_to<transfer_sender_t> _Self, class _Env>
      friend auto tag_invoke(std::execution::get_completion_signatures_t, _Self&&, _Env) ->
        std::execution::make_completion_signatures<
          std::__member_t<_Self, Sender>,
          _Env> requires true;

    template <std::execution::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
      requires std::__callable<Tag, const Sender&, As...>
    friend auto tag_invoke(Tag tag, const transfer_sender_t& self, As&&... as)
      noexcept(std::__nothrow_callable<Tag, const Sender&, As...>)
      -> std::__call_result_if_t<std::execution::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
      return ((Tag&&) tag)(self.sndr_, (As&&) as...);
    }

    transfer_sender_t(detail::queue::task_hub_t* hub, Sender sndr)
      : hub_(hub)
      , sndr_{(Sender&&)sndr} {
    }
  };

}
