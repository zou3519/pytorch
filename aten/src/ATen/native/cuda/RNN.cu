#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Error.h"

#include <cfloat>
#include <tuple>

namespace at {
namespace native {

__forceinline__ __device__ float sigmoidf(float in) {
   return 1.f / (1.f + expf(-in));
}

// Fused forward kernel
__global__ void lstm_kernel(int hiddenSize, int miniBatch,
                            const float *igates,
                            const float *hgates,// batchSize, 4 * hiddenSize
                            const float *ibias, // 4 * hiddenSize
                            const float *hbias,
                            const float *c_in,  // batch * hiddenSize
                            //float *linearGates,
                            float *h_out,       // batch * hiddenSize
                            float *c_out
                            //bool training) {
                            ) {
   int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
   int numElements = miniBatch * hiddenSize;  // # of output elements

   if (linearIndex >= numElements) return;

   // h_out[batch][index]
   // assuming contiguous input
   int batch = linearIndex / hiddenSize;
   int index = linearIndex % hiddenSize;

   int gateIndex = index + 4 * batch * hiddenSize;

   float g[4];

   for (int i = 0; i < 4; i++) {
      g[i] = igates[i * hiddenSize + gateIndex] + hgates[i * hiddenSize + gateIndex];
      g[i] += ibias[i * hiddenSize + index] + hbias[i * hiddenSize + index];

      // if (training) linearGates[gateIndex + i * hiddenSize] = g[i];
   }

   float in_gate     = sigmoidf(g[0]);
   float forget_gate = sigmoidf(g[1]);
   float in_gate2    = tanhf(g[2]);
   float out_gate    = sigmoidf(g[3]);

   float val = (forget_gate * c_in[linearIndex]) + (in_gate * in_gate2);

   c_out[linearIndex] = val;

   val = out_gate * tanhf(val);

   h_out[linearIndex] = val;
   // i_out[linearIndex] = val;
}

void lstm_fused_kernel(
    const Tensor& igates,
    const Tensor& hgates,
    const Tensor& ibias,
    const Tensor& hbias,
    const Tensor& c_in,
    Tensor& h_out,
    Tensor& c_out)
{
  int hiddenSize = c_in.size(1);
  int miniBatch = c_in.size(0);
  int numElements = c_in.numel();
  
  dim3 blockDim;
  dim3 gridDim;
  blockDim.x = 256;
  gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

  lstm_kernel <<< gridDim, blockDim, 0, globalContext().getCurrentCUDAStream() >>>
      (hiddenSize, miniBatch,
       igates.data<float>(),
       hgates.data<float>(),
       ibias.data<float>(),
       hbias.data<float>(),
       c_in.data<float>(),
       h_out.data<float>(),
       c_out.data<float>());
}

std::tuple<Tensor,Tensor> lstm_chunk_pw_fused_cuda(
    const Tensor& self,
    const Tensor& hgates,
    const Tensor& ibias,
    const Tensor& hbias,
    const Tensor& c_in)
{
  auto batch_size = self.size(0);
  auto hidden_size = self.size(1) / 4;  // hidden_size == input_size

  auto hx_out = self.type().empty({batch_size, hidden_size});
  auto cx_out = self.type().empty({batch_size, hidden_size});
  lstm_fused_kernel(self, hgates, ibias, hbias, c_in, hx_out, cx_out);
  return std::tuple<Tensor,Tensor>(hx_out, cx_out);
}

std::tuple<Tensor, Tensor, Tensor> lstm_fusion_prealloc(
    const Tensor& self,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& w_ih,
    const Tensor& w_hh,
    const Tensor& b_ih,
    const Tensor& b_hh) {
  auto seq_len = self.size(0);
  auto batch_size = self.size(1);
  auto hidden_size = self.size(2);  // hidden_size == input_size

  auto w_ih_t = w_ih.t().contiguous();
  auto w_hh_t = w_hh.t().contiguous();

  // workspace
  auto igates = self.type().empty({seq_len, batch_size, 4 * hidden_size});
  auto hgates = self.type().empty({seq_len, batch_size, 4 * hidden_size});

  auto hy = hx.select(0, 0);
  auto cy = cx.select(0, 0);

  auto hx_out = self.type().empty({seq_len, batch_size, hidden_size});
  auto cx_out = self.type().empty({seq_len, batch_size, hidden_size});


  for (int64_t i = 0; i < seq_len; i++) {
    auto input_ = self.select(0, i);
    auto igate_ = igates.select(0, i);
    auto hgate_ = hgates.select(0, i);

    auto hx_next = hx_out.select(0, i);
    auto cx_next = cx_out.select(0, i);

    at::mm_out(igate_, input_, w_ih_t);
    at::mm_out(hgate_, hy, w_hh_t);
    lstm_fused_kernel(igate_, hgate_, b_ih, b_hh, cy,
                      hx_next, cx_next);

    hy = hx_next;
    cy = cx_next;
  }

  return std::tuple<Tensor,Tensor,Tensor>(
      hx_out, hx_out.narrow(0, seq_len-1, 1).clone(), cx_out.narrow(0, seq_len-1, 1).clone());
}

std::tuple<Tensor, Tensor, Tensor> lstm_fusion_no_prealloc(
    const Tensor& self,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& w_ih,
    const Tensor& w_hh,
    const Tensor& b_ih,
    const Tensor& b_hh) {
  auto seq_len = self.size(0);
  auto batch_size = self.size(1);
  auto hidden_size = self.size(2);  // hidden_size == input_size

  auto w_ih_t = w_ih.t().contiguous();
  auto w_hh_t = w_hh.t().contiguous();

  auto hy = hx.select(0, 0);
  auto cy = cx.select(0, 0);

  std::vector<Tensor> outputs(seq_len);

  for (int64_t i = 0; i < seq_len; i++) {
    auto input_ = self.select(0, i);

    auto hx_next = at::empty_like(hy);
    auto cx_next = at::empty_like(cx);

    auto igate = at::mm(input_, w_ih_t);
    auto hgate = at::mm(hy, w_hh_t);
    lstm_fused_kernel(igate, hgate, b_ih, b_hh, cy,
                      hx_next, cx_next);

    hy = hx_next;
    cy = cx_next;
    outputs[i] = hy;
  }

  auto output = at::stack(outputs, 0);

  return std::tuple<Tensor,Tensor,Tensor>(
      output, hy.unsqueeze(0), cy.unsqueeze(0));
}


std::tuple<Tensor, Tensor, Tensor> lstm_fusion_cuda(
    const Tensor& self,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& w_ih,
    const Tensor& w_hh,
    const Tensor& b_ih,
    const Tensor& b_hh,
    int64_t choice) {
  if (choice == 0) {
    return lstm_fusion_prealloc(self, hx, cx, w_ih, w_hh, b_ih, b_hh);
  } else if (choice == 1) {
    return lstm_fusion_no_prealloc(self, hx, cx, w_ih, w_hh, b_ih, b_hh);
  }
  throw std::runtime_error("wat");
}

}}
