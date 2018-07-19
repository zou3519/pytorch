#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> lstm_fusion_cpu(
    const Tensor& self,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& w_ih,
    const Tensor& w_hh,
    const Tensor& b_ih,
    const Tensor& b_hh,
    int64_t choice) {
  throw std::runtime_error("Not Implemented");
}

std::tuple<Tensor,Tensor> lstm_chunk_pw_fused_cpu(
    const Tensor& self,
    const Tensor& hgates,
    const Tensor& ibias,
    const Tensor& hbias,
    const Tensor& c_in)
{
  throw std::runtime_error("NOT implemented");
}

static std::tuple<Tensor,Tensor> lstm_cell(
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& w_ih,
    const Tensor& w_hh,
    const Tensor& b_ih,
    const Tensor& b_hh) {
  // assumes pre-transposed inputs
  auto igate = at::mm(input, w_ih);
  auto hgate = at::mm(hx, w_hh);

  auto t9 = igate + hgate + b_ih + b_hh;

  auto r0 = at::chunk(t9, 4, 1);
  auto t11 = r0[0];
  auto t12 = r0[1];
  auto t13 = r0[2];
  auto t14 = r0[3];
  
  auto t15 = at::sigmoid(t11);
  auto t16 = at::sigmoid(t12);
  auto t17 = at::tanh(t13);
  auto t18 = at::sigmoid(t14);
  auto t19 = at::mul(t16, cx);
  auto t21 = at::mul(t15, t17);
  auto t22 = at::add(t19, t21);
  auto t23 = at::tanh(t22)   ;
  auto t24 = at::mul(t18, t23);
  // returns cy, hy
  return std::tuple<Tensor,Tensor>(t22, t24);         
}

std::tuple<Tensor,Tensor,Tensor> lstm_aten(
    const Tensor& self,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& w_ih,
    const Tensor& w_hh,
    const Tensor& b_ih,
    const Tensor& b_hh) {
  //auto seq_length = self.size(0);
  //Tensor hx_tmp, cx_tmp;

  //auto hy = hx[0];
  //auto cy = cx[0]; 

  //std::vector<Tensor> outputs(seq_length);

  //for (int index = 0; index < seq_length; index++) {
  //  hx_tmp = hy;
  //  cx_tmp = cy;

  //  auto input = self[index];
  //  auto igates = at::mm(input, w_ih);
  //  auto hgates = at::mm(hx_tmp, w_hh);
  //  at::lstm_fused_pointwise(igates, hgates, b_ih, b_hh, cx_tmp, hy, cy);
  //  outputs.push_back(hy);
  //}
  //
  // auto y = at::stack(outputs, 0);
  // hy = hy.unsqueeze(0);
  // cy = cy.unsqueeze(0);
  // return std::tuple<Tensor,Tensor,Tensor>(y, hy, cy);

  // assumes pre-transposed inputs
  auto t8 = at::select(self, 0, 0);
  Tensor t139, t140;
  std::tie(t139, t140) = lstm_cell(t8, hx, cx, w_ih, w_hh, b_ih, b_hh);

  auto t31  = at::select(self, 0, 1);
  Tensor t141, t142;
  std::tie(t141, t142) = lstm_cell(t31, t140, t139, w_ih, w_hh, b_ih, b_hh);

  auto t54  = at::select(self, 0, 2);
  Tensor t143, t144;
  std::tie(t143, t144) = lstm_cell(t54, t142, t141, w_ih, w_hh, b_ih, b_hh);

  auto t77  = at::select(self, 0, 3);
  Tensor t145, t146;
  std::tie(t145, t146) = lstm_cell(t77, t144, t143, w_ih, w_hh, b_ih, b_hh);

  auto t100  = at::select(self, 0, 4);
  Tensor t147, t148;
  std::tie(t147, t148) = lstm_cell(t100, t146, t145, w_ih, w_hh, b_ih, b_hh);

  auto t123 = at::cat({t140, t142, t144, t146, t148}, 0);
  auto t124 = at::size(self, 0);
  auto t125 = at::size(t140, 0);
  auto t126 = at::size(t140, 1);
  auto t128 = t123.view({t124, t125, t126});

  auto t129 = t148.size(0);
  auto t130 = t148.size(1);
  auto t131 = 1;
  auto t133 = t148.view({t131, t129, t130});

  auto t134 = at::size(t147, 0);
  auto t135 = at::size(t147, 1);
  auto t138 = t147.view({t131, t134, t135});
  return std::tuple<Tensor,Tensor,Tensor>(t128, t133, t138);
}


std::tuple<Tensor, Tensor, Tensor> lstm_fusion_no_prealloc_native(
    const Tensor& self,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& w_ih,
    const Tensor& w_hh,
    const Tensor& b_ih,
    const Tensor& b_hh) {
  auto seq_len = self.size(0);

  auto w_ih_t = w_ih.t().contiguous();
  auto w_hh_t = w_hh.t().contiguous();

  auto hy = hx.select(0, 0);
  auto cy = cx.select(0, 0);

  std::vector<Tensor> outputs(seq_len);

  for (int64_t i = 0; i < seq_len; i++) {
    auto input_ = self.select(0, i);

    auto igate = at::mm(input_, w_ih_t);
    auto hgate = at::mm(hy, w_hh_t);
    std::tie(hy, cy) = lstm_chunk_pw_fused(igate, hgate, b_ih, b_hh, cy);

    outputs[i] = hy;
  }

  auto output = at::stack(outputs, 0);

  return std::tuple<Tensor,Tensor,Tensor>(
      output, hy.unsqueeze(0), cy.unsqueeze(0));
}

std::tuple<Tensor, Tensor, Tensor> lstm_native(
    const Tensor& self,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& w_ih,
    const Tensor& w_hh,
    const Tensor& b_ih,
    const Tensor& b_hh) {
  return lstm_fusion_no_prealloc_native(
      self, hx, cx, w_ih, w_hh, b_ih, b_hh);
}

}}
