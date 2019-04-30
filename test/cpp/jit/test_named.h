#pragma once

#include "test/cpp/jit/test_base.h"
#include <torch/csrc/autograd/named.h>

namespace torch {
namespace jit {

using namespace torch::autograd::named;  // TODO: namespacing

void testNamed_valid_names() {
  auto result = torch::autograd::named::check_valid_name("N");
  ASSERT_EQ(result.first, torch::autograd::named::NameType::NORMAL);
  ASSERT_FALSE(result.second);

  result = torch::autograd::named::check_valid_name("batch");
  ASSERT_EQ(result.first, torch::autograd::named::NameType::NORMAL);
  ASSERT_FALSE(result.second);

  result = torch::autograd::named::check_valid_name("C.in");
  ASSERT_EQ(result.first, torch::autograd::named::NameType::TAGGED);
  ASSERT_EQ(result.second.value(), "C");

  result = torch::autograd::named::check_valid_name("foo.bar");
  ASSERT_EQ(result.first, torch::autograd::named::NameType::TAGGED);
  ASSERT_EQ(result.second.value(), "foo");

  ASSERT_THROWS_WITH(torch::autograd::named::check_valid_name(""), "Invalid name");
  ASSERT_THROWS_WITH(torch::autograd::named::check_valid_name(".bar"), "Invalid name");
  ASSERT_THROWS_WITH(torch::autograd::named::check_valid_name("."), "Invalid name");
  ASSERT_THROWS_WITH(torch::autograd::named::check_valid_name("bar."), "Invalid name");
}

static void check_unify(DimName dimname, DimName other, c10::optional<DimName> expected) {
  auto result = torch::autograd::named::unify(dimname, other);
  if (expected) {
    ASSERT_EQ(result->name(), expected->name());
    ASSERT_EQ(result->type(), expected->type());
    ASSERT_EQ(result->basename(), expected->basename());
  } else {
    ASSERT_FALSE(result);
  }
}

void testNamed_unify() {
  auto wildcard = DimName();
  auto idA = reinterpret_cast<InternedName>((uint64_t)1);
  auto idB = reinterpret_cast<InternedName>((uint64_t)2);
  auto idC = reinterpret_cast<InternedName>((uint64_t)3);
  auto idD = reinterpret_cast<InternedName>((uint64_t)4);

  check_unify(DimName(idA), DimName(idA), DimName(idA));
  check_unify(DimName(idA), wildcard, DimName(idA));
  check_unify(wildcard, DimName(idA), DimName(idA));
  check_unify(wildcard, wildcard, wildcard);

  check_unify(DimName(idA), DimName(idB), c10::nullopt);

  // DimName(idA, idB) = "B.<something>"
  check_unify(wildcard, DimName(idA, idB), DimName(idA, idB));
  check_unify(DimName(idB), DimName(idA, idB), DimName(idB));
  check_unify(DimName(idC), DimName(idA, idB), c10::nullopt);
  check_unify(DimName(idC, idB), DimName(idA, idB), DimName(idB));
  check_unify(DimName(idA, idB), DimName(idA, idB), DimName(idA, idB));
  check_unify(DimName(idB, idA), DimName(idD, idC), c10::nullopt);
}

}}
