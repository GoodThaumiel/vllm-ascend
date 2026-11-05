// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>

#include "ops.h"

TORCH_LIBRARY(replayssm_sd_310p, m) {
  m.def("io_probe(Tensor input, Tensor(a!) output, int mode) -> ()");
  m.def("verify_(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor checkpoint, "
        "Tensor(a!) d_cache, Tensor(b!) k_cache, Tensor(c!) g_cache, Tensor slots, "
        "Tensor positions, Tensor bases, Tensor lengths, Tensor(d!) out, Tensor(e!) status, "
        "float scale, int tile_v=32, int cores=7, Tensor? ring_slots=None) -> ()");
  m.def("compact_(Tensor(a!) checkpoint, Tensor d_cache, Tensor k_cache, Tensor g_cache, "
        "Tensor slots, Tensor positions, Tensor bases, Tensor flush, Tensor(b!) status, "
        "int tile_v=32, int cores=7, Tensor? ring_slots=None) -> ()");
  m.def("cursor_(Tensor(a!) write_pos, Tensor(b!) cache_base, Tensor(c!) is_flush, "
        "Tensor accepted, Tensor slots, Tensor do_reset, Tensor(d!) status, "
        "int logical_limit=16, int max_spec_len=3, int ring_len=32) -> ()");
}

TORCH_LIBRARY_IMPL(replayssm_sd_310p, PrivateUse1, m) {
  m.impl("io_probe", &replayssm_sd_310p::io_probe);
  m.impl("verify_", &replayssm_sd_310p::verify);
  m.impl("compact_", &replayssm_sd_310p::compact);
  m.impl("cursor_", &replayssm_sd_310p::cursor);
}

TORCH_LIBRARY_IMPL(replayssm_sd_310p, Meta, m) {
  m.impl("io_probe", [](const at::Tensor&, const at::Tensor&, int64_t) {});
  m.impl("verify_", [](const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        double, int64_t, int64_t, c10::optional<at::Tensor>) {});
  m.impl("compact_", [](const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         int64_t, int64_t,
                         c10::optional<at::Tensor>) {});
  m.impl("cursor_", [](const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, int64_t, int64_t, int64_t) {});
}
