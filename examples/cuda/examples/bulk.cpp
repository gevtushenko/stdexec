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
#include "schedulers/stream.hpp"

namespace ex = std::execution;

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

struct printer_t {
  int val_;

  __host__ __device__ void operator()(int idx, int val) {
    printf("%d: %d (on %s) val = %d\n", val_, idx, (is_on_gpu() ? "gpu" : "cpu"), val);
  }
};

int main() {
  example::cuda::stream::scheduler_t scheduler{};

  auto snd = ex::schedule(scheduler) 
           | ex::then([]() -> int { return 42; })
           | ex::bulk(4, printer_t{1})
           | ex::bulk(4, printer_t{2});
  std::this_thread::sync_wait(std::move(snd));
}

