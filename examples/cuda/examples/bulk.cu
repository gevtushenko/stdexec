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

#include <execution.hpp>
#include "schedulers/stream.cuh"
#include <thrust/device_vector.h>

namespace ex = std::execution;
using example::cuda::is_on_gpu;

struct printer_t {
  int val_;

  __host__ __device__ void operator()(int idx, int val) {
    printf("%d: %d (on %s) val = %d\n", val_, idx, (is_on_gpu() ? "gpu" : "cpu"), val);
  }
};

__host__ __device__ 
std::uint64_t fib(std::uint64_t n) {
   if (n <= 1) {
     return n;
   }

   return fib(n - 1) + fib(n - 2);
}

int main() {
  example::cuda::stream::scheduler_t scheduler{};

  auto snd = ex::schedule(scheduler) 
           | ex::then([]() { std::printf("+1\n"); })
           | ex::bulk(42, [](int idx) {});
  std::this_thread::sync_wait(std::move(snd));
}

