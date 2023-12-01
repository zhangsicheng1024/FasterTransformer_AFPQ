/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "3rdparty/INIReader.h"
#include "examples/cpp/multi_gpu_gpt/gpt_example_utils.h"
#include "src/fastertransformer/models/llama/Llama.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"
#include "src/fastertransformer/utils/word_list.h"
#include "src/fastertransformer/utils/ladder_gemm/ladder_kernel.h"

#include <cuda_profiler_api.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

using namespace fastertransformer;

void llama_gemm(int M, int K, int N);

int main(int argc, char* argv[])
{
    int M = std::stoi(argv[1]);
    int K = std::stoi(argv[2]);
    int N = std::stoi(argv[3]);

    llama_gemm(M, K, N);

    return 0;
}

void llama_gemm(int M, int K, int N)
{
    half* A;
    half* B;
    half* C;
    deviceMalloc(&A, M * K, false);
    deviceMalloc(&B, K * N, false);
    deviceMalloc(&C, M * N, false);

    loadWeightFromBin<half>(B, {N * K / 4 + K / 128 * N}, "/data/llama-2-7b-chat-int4-ft/1-gpu/model.layers.0.attention.dense.weight.0.bin", FtCudaDataType::FP16);
    // loadWeightFromBin<half>(B, {N * K / 4 + K / 128 * N}, "/data/debug/weights.bin", FtCudaDataType::FP16);
    loadWeightFromBin<half>(A, {M, K}, "/data/debug/inputs.bin", FtCudaDataType::FP16);

    // printMatrix(B + 8388608 / 2, K / 128, N, N, true);

    ladder_gemm_fp16xnf4_fp16(A, B, C, M, N, K, 0, 1);

    saveToBinary<half>(C, M * N, "/data/debug/outputs_ft.bin");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return;
}
