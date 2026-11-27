"""vae_engine.CarryStore (DESIGN §9.43): a video stripe's carries live in one buffer.

After each call_unit the engine copies every new device tensor of the stripe's state into the unit's part of the
buffer and puts the view in its place; views the state still holds stay where they are, a tensor that does not fit
the part stays where it was. Plain CPU tensors, a fake adapter: no model, no GPU.
    python tests/test_vae_carry.py
"""

import types

from common import check, finish
import torch

from monoload import vae_engine as eng

MIB = 1 << 20


def main():
    # two units; unit 0 keeps 2 carries of 1 MiB (its part 2 MiB), unit 1 one of 3 MiB
    units = [types.SimpleNamespace(name="u0", k=2), types.SimpleNamespace(name="u1", k=3)]
    bound = types.SimpleNamespace(name="fake", unit_cache=lambda u, r, w, e: u.k * r * w * e)
    plan = types.SimpleNamespace(widths=[256, 256, 256], elem=4, needs=[[(0, 512), (0, 1024), (0, 1024)], [(0, 256), (0, 512), (0, 512)]])
    st = eng.CarryStore.make(bound, units, plan, torch.device("cpu"))
    check("buffer = the largest stripe's carries (2 + 4 MiB)", st.buf.numel() == 6 * MIB, repr(st.buf.numel()))
    st.stripe(plan.needs[0])
    check("unit parts follow unit_cache", st.parts == [(0, 2 * MIB), (2 * MIB, 6 * MIB)], repr(st.parts))

    def mb(n, v):
        return torch.full((n * MIB // 4,), float(v))

    big = mb(4, 7)
    state = {"u0": [mb(1, 1), big[: MIB // 4]], "idx": [0], "small": torch.ones(8)}
    st.settle(state, 0)
    a, b = state["u0"]
    key = st.buf.untyped_storage()._cdata
    check("new carries moved into the buffer", a.untyped_storage()._cdata == key and b.untyped_storage()._cdata == key)
    check("values kept", bool((a == 1).all()) and bool((b == 7).all()))
    check("packed in unit 0's part", st._span(a) == (0, MIB) and st._span(b) == (MIB, 2 * MIB), repr((st._span(a), st._span(b))))
    check("a slice of a larger tensor no longer keeps it alive", b.untyped_storage()._cdata != big.untyped_storage()._cdata)
    check("small tensors and non-tensors left alone", state["small"].numel() == 8 and state["idx"] == [0])

    # next chunk: unit 0 replaces one carry and keeps the other -> the kept view stays, the new one takes the free gap
    keep = state["u0"][1]
    state["u0"] = [mb(1, 2), keep]
    st.settle(state, 0)
    a, b = state["u0"]
    check("kept view untouched", b is keep and bool((b == 7).all()))
    check("replacement reuses the dead slot", st._span(a) == (0, MIB) and bool((a == 2).all()), repr(st._span(a)))

    # unit 1: too large for its part -> stays where it is
    huge = mb(5, 3)
    state["u1"] = huge
    st.settle(state, 1)
    check("a carry larger than the unit's part stays (counted)", state["u1"] is huge and st.left == 1, repr(st.left))
    state["u1"] = mb(3, 4)
    st.settle(state, 1)
    check("unit 1's carry in its own part", st._span(state["u1"]) == (2 * MIB, 5 * MIB), repr(st._span(state["u1"])))
    check("unit 0's views not overwritten", bool((state["u0"][0] == 2).all()) and bool((state["u0"][1] == 7).all()))

    # a new stripe frees everything
    st.stripe(plan.needs[1])
    state = {"u1": mb(1, 5)}
    st.settle(state, 1)
    check("new stripe: smaller parts, carry at its part's start", st.parts[1][0] == MIB and st._span(state["u1"]) == (MIB, 2 * MIB),
          repr((st.parts, st._span(state["u1"]))))

    # nested containers, a non-contiguous carry
    st.stripe(plan.needs[0])
    x = torch.arange(4 * MIB // 4, dtype=torch.float32).view(1024, 1024)
    nc = x[:, :256]   # 1 MiB, rows strided
    state = {"u0": {"deep": [nc]}}
    st.settle(state, 0)
    got = state["u0"]["deep"][0]
    check("nested, non-contiguous carry moved with its values", got.untyped_storage()._cdata == key and torch.equal(got, nc))

    # an adapter without carries: no store
    check("no unit_cache: no store", eng.CarryStore.make(types.SimpleNamespace(name="x"), units, plan, torch.device("cpu")) is None)
    zero = types.SimpleNamespace(name="z", unit_cache=lambda u, r, w, e: 0)
    check("unit_cache 0: no store", eng.CarryStore.make(zero, units, plan, torch.device("cpu")) is None)
    finish()


if __name__ == "__main__":
    main()
