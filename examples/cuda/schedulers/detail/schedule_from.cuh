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

namespace schedule_from {

template <class ReceiverId>
  class receiver_t
    : std::execution::receiver_adaptor<receiver_t<ReceiverId>, std::__t<ReceiverId>>
    , receiver_base_t {
    using Receiver = std::__t<ReceiverId>;
    friend std::execution::receiver_adaptor<receiver_t, Receiver>;

    operation_state_base_t &op_state_;

    template <class... As>
    void set_value(As&&... as) && noexcept {
      // TODO Wrap random receiver so it's completed on GPU:
      // `transfer(stream) | a_sender`
      std::execution::set_value(std::move(this->base()), (As&&)as...);
    }

   public:
    explicit receiver_t(Receiver rcvr, operation_state_base_t &op_state)
      : std::execution::receiver_adaptor<receiver_t, Receiver>((Receiver&&) rcvr)
      , op_state_(op_state)
    {}
  };

}

template <class Scheduler, class SenderId>
  struct schedule_from_sender_t {
    using Sender = std::__t<SenderId>;
    Sender sndr_;

    template <class Receiver>
      using receiver_t = schedule_from::receiver_t<std::__x<Receiver>>;

    template <std::__decays_to<schedule_from_sender_t> Self, std::execution::receiver Receiver>
      requires std::execution::sender_to<std::__member_t<Self, Sender>, Receiver>
    friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
      -> stream_op_state_t<std::__member_t<Self, Sender>, receiver_t<Receiver>> {
        return stream_op_state<std::__member_t<Self, Sender>>(((Self&&)self).sndr_, [&](operation_state_base_t& stream_provider) -> receiver_t<Receiver> {
          return receiver_t<Receiver>((Receiver&&)rcvr, stream_provider);
        });
    }

    template <std::__one_of<std::execution::set_value_t, std::execution::set_stopped_t> _Tag>
    friend Scheduler tag_invoke(std::execution::get_completion_scheduler_t<_Tag>, const schedule_from_sender_t& __self) noexcept {
      return {};
    }

    template <std::execution::tag_category<std::execution::forwarding_sender_query> _Tag, class... _As _NVCXX_CAPTURE_PACK(_As)>
      requires std::__callable<_Tag, const Sender&, _As...>
    friend auto tag_invoke(_Tag __tag, const schedule_from_sender_t& __self, _As&&... __as)
      noexcept(std::__nothrow_callable<_Tag, const Sender&, _As...>)
      -> std::__call_result_if_t<std::execution::tag_category<_Tag, std::execution::forwarding_sender_query>, _Tag, const Sender&, _As...> {
      _NVCXX_EXPAND_PACK_RETURN(_As, _as,
        return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
      )
    }

    template <std::__decays_to<schedule_from_sender_t> _Self, class _Env>
      friend auto tag_invoke(std::execution::get_completion_signatures_t, _Self&&, _Env) ->
        std::execution::make_completion_signatures<
          std::__member_t<_Self, Sender>,
          _Env>;
  };

}
