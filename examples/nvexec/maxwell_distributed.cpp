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

#include "exec/inline_scheduler.hpp"
#include "nvexec/stream_context.cuh"

namespace ex = stdexec;

void f() {}

int main(int argc, char *argv[]) {
  nvexec::stream_context ctx{};
  auto gpu = ctx.get_scheduler();

  auto compute_h = ex::just() 
                 | ex::transfer(gpu) 
                 | ex::then(f) 
                 | ex::transfer(exec::inline_scheduler{})
                 | ex::transfer(gpu)
                 | ex::then(f)
                 | ex::transfer(exec::inline_scheduler{});

  stdexec::this_thread::sync_wait(std::move(compute_h)); 
}

