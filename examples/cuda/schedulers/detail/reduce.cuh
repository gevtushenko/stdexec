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

#include <cuda/std/type_traits>

#include <cub/device/device_reduce.cuh>

#include "common.cuh"

namespace example::cuda::stream {

namespace reduce_ {
  template <typename Invokable, typename InputT>
    using accumulator_t =
      std::add_lvalue_reference_t<
        ::cuda::std::decay_t<
          ::cuda::std::invoke_result_t<Invokable, InputT, InputT>
        >
      >;

  template <class ReceiverId, class IteratorId, class Fun>
    class receiver_t : receiver_base_t {
      using Receiver = std::__t<ReceiverId>;
      using Iterator = std::__t<IteratorId>;
      using Result = accumulator_t<Fun, typename std::iterator_traits<Iterator>::value_type>;

      Iterator d_in_;
      std::size_t num_items_;
      Fun f_;
      operation_state_base_t<ReceiverId> &op_state_;

    public:

      friend void tag_invoke(std::execution::set_value_t, receiver_t&& self) noexcept {
        cudaStream_t stream = self.op_state_.stream_;

        using value_t = std::decay_t<Result>;
        value_t *d_out{};
        THROW_ON_CUDA_ERROR(cudaMallocAsync(&d_out, sizeof(value_t), stream));

        void *d_temp_storage{};
        std::size_t temp_storage_size{};

        THROW_ON_CUDA_ERROR(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_size, self.d_in_,
                                                      d_out, self.num_items_, self.f_, value_t{},
                                                      stream));
        THROW_ON_CUDA_ERROR(cudaMallocAsync(&d_temp_storage, temp_storage_size, stream));

        THROW_ON_CUDA_ERROR(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_size, self.d_in_,
                                                      d_out, self.num_items_, self.f_, value_t{},
                                                      stream));
        THROW_ON_CUDA_ERROR(cudaFreeAsync(d_temp_storage, stream));

        self.op_state_.propagate_completion_signal(std::execution::set_value, *d_out);
        THROW_ON_CUDA_ERROR(cudaFreeAsync(d_out, stream));
      }

      template <std::__one_of<std::execution::set_error_t,
                              std::execution::set_stopped_t> Tag,
                class... As _NVCXX_CAPTURE_PACK(As)>
        friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
          _NVCXX_EXPAND_PACK(As, as,
            self.op_state_.propagate_completion_signal(tag, (As&&)as...);
          );
        }

      friend std::execution::env_of_t<Receiver> tag_invoke(std::execution::get_env_t, const receiver_t& self) {
        return std::execution::get_env(self.op_state_.receiver_);
      }

      receiver_t(Iterator it, std::size_t num_items, Fun fun, operation_state_base_t<ReceiverId> &op_state)
        : d_in_(it)
        , num_items_(num_items)
        , f_((Fun&&) fun)
        , op_state_(op_state)
      {}
    };
}

template <class SenderId, class IteratorId, class FunId>
  struct reduce_sender_t : gpu_sender_base_t {
    using Sender = std::__t<SenderId>;
    using Iterator = std::__t<IteratorId>;
    using Fun = std::__t<FunId>;
    using Result = reduce_::accumulator_t<Fun, typename std::iterator_traits<Iterator>::value_type>;

    Sender sndr_;
    Iterator it_;
    std::size_t num_items_;
    Fun fun_;

    template <class Receiver>
      using receiver_t = reduce_::receiver_t<std::__x<Receiver>, IteratorId, Fun>;

    template <class... Ts>
      using set_value_t =
        std::execution::completion_signatures<std::execution::set_value_t(Result)>;

    template <class Self, class Env>
      using completion_signatures =
        std::execution::make_completion_signatures<
          std::__member_t<Self, Sender>,
          Env,
          std::execution::completion_signatures<std::execution::set_error_t(cudaGraph_t)>,
          set_value_t>;

    template <std::__decays_to<reduce_sender_t> Self, std::execution::receiver Receiver>
      requires std::execution::receiver_of<Receiver, completion_signatures<Self, std::execution::env_of_t<Receiver>>>
    friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
      -> stream_op_state_t<std::__member_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<std::__member_t<Self, Sender>>(
          ((Self&&)self).sndr_,
          (Receiver&&)rcvr,
          [&](operation_state_base_t<std::__x<Receiver>>& stream_provider) -> receiver_t<Receiver> {
            return receiver_t<Receiver>(self.it_, self.num_items_, self.fun_, stream_provider);
          });
    }

    template <std::__decays_to<reduce_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> std::execution::dependent_completion_signatures<Env>;

    template <std::__decays_to<reduce_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> completion_signatures<Self, Env> requires true;

    template <std::execution::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
      requires std::__callable<Tag, const Sender&, As...>
    friend auto tag_invoke(Tag tag, const reduce_sender_t& self, As&&... as)
      noexcept(std::__nothrow_callable<Tag, const Sender&, As...>)
      -> std::__call_result_if_t<std::execution::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
      return ((Tag&&) tag)(self.sndr_, (As&&) as...);
    }
  };

struct reduce_t {
  template <class Sender, class Iterator, class Fun>
    using __sender =
      reduce_sender_t<
        std::__x<std::remove_cvref_t<Sender>>,
        std::__x<std::remove_cvref_t<Iterator>>,
        std::__x<std::remove_cvref_t<Fun>>>;

  template <std::execution::sender Sender, class Iterator, std::execution::__movable_value Fun>
    __sender<Sender, Iterator, Fun> operator()(Sender&& __sndr, Iterator it, std::size_t num_items, Fun __fun) const {
      return __sender<Sender, Iterator, Fun>{{}, (Sender&&) __sndr, it, num_items, (Fun&&) __fun};
    }

  template <class Iterator, class Fun = cub::Sum>
    std::execution::__binder_back<reduce_t, Iterator, std::size_t, Fun> operator()(Iterator it, std::size_t num_items, Fun __fun={}) const {
      return {{}, {}, {it, num_items, (Fun&&) __fun}};
    }
};

inline constexpr reduce_t reduce{};
}

