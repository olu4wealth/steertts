#!/usr/bin/env python3
"""Builds the stub ONNX graphs used by StubGraphInferenceTest.

The real SteerTTS weights are licence-gated (CC BY-NC 4.0), so the JVM test cannot run them.
Instead these stubs declare the exact input signature the engine probes for and echo every
steering input back as their "audio" output. That makes the engine's tensor wiring assertable:

    audio[1,132] = ctrl(12) | sum(cmask) | n | tag | sum(spk) | 116 zeros

Requires: pip install onnx numpy  (numpy + onnx only; onnxruntime optional, for a quick check)

Usage:  python tools/gen_stub_models.py [--check]
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h, numpy_helper

OUT = "app/src/test/resources/steer"


def _fi(text):
    return h.make_tensor_value_info(text, T.FLOAT, [1, 12])


def _vi(text, dtype, shape):
    return h.make_tensor_value_info(text, dtype, shape)


def echo():
    """Echoes steering back out as a 132-sample 'waveform' (>= the 64-sample minimum)."""
    nodes = [
        # rank-2 reductions: cmask over both axes, spk over the embedding axis
        h.make_node("ReduceSum", ["cmask", "ax_all"], ["mask_sum"], keepdims=1),   # [1,1]
        h.make_node("ReduceSum", ["spk", "ax_1"], ["spk_sum"], keepdims=1),        # [1,1]
        # n and tag arrive as rank-1 [1] tensors, so lift them to [1,1] for Concat
        h.make_node("Cast", ["n"], ["n_f"], to=T.FLOAT),
        h.make_node("Cast", ["tag"], ["tag_f"], to=T.FLOAT),
        h.make_node("Unsqueeze", ["n_f", "ax_0"], ["n_u"]),                        # [1,1]
        h.make_node("Unsqueeze", ["tag_f", "ax_0"], ["tag_u"]),                    # [1,1]
        h.make_node(
            "Concat",
            ["ctrl", "mask_sum", "n_u", "tag_u", "spk_sum", "zeros_tail"],
            ["audio"],
            axis=1,
        ),                                                                         # [1,132]
    ]
    inits = [
        numpy_helper.from_array(np.array([0, 1], np.int64), "ax_all"),
        numpy_helper.from_array(np.array([1], np.int64), "ax_1"),
        numpy_helper.from_array(np.array([0], np.int64), "ax_0"),
        numpy_helper.from_array(np.zeros((1, 116), np.float32), "zeros_tail"),
    ]
    graph = h.make_graph(
        nodes,
        "steer_echo",
        [
            _vi("ids", T.INT64, [1, "N"]),
            _vi("n", T.INT64, [1]),
            _vi("spk", T.FLOAT, [1, 8]),
            _vi("tag", T.INT64, [1]),
            _vi("ctrl", T.FLOAT, [1, 12]),
            _vi("cmask", T.FLOAT, [1, 12]),
        ],
        [_vi("audio", T.FLOAT, [1, 132])],
        initializer=inits,
    )
    return h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], producer_name="stub")


def narrow():
    """A pooled-feature output [1,4] - the shape an unfused enc/dec/voc graph exposes."""
    nodes = [h.make_node("Identity", ["small"], ["audio"])]
    inits = [numpy_helper.from_array(np.zeros((1, 4), np.float32), "small")]
    graph = h.make_graph(
        nodes,
        "steer_narrow",
        [_vi("ids", T.INT64, [1, "N"]), _vi("ctrl", T.FLOAT, [1, 12])],
        [_vi("audio", T.FLOAT, [1, 4])],
        initializer=inits,
    )
    return h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], producer_name="stub")


def main():
    for name, model in (("echo", echo()), ("narrow", narrow())):
        onnx.checker.check_model(model)
        # onnx >= 1.20 writes IR 14; onnxruntime 1.31.1 refuses anything above IR 13.
        model.ir_version = 10
        path = f"{OUT}/{name}.onnx"
        onnx.save(model, path)
        print(f"{path}: ok, {len(model.SerializeToString())} bytes, ir={model.ir_version}")

    if "--check" in sys.argv:
        import onnxruntime as ort

        session = ort.InferenceSession(f"{OUT}/echo.onnx", providers=["CPUExecutionProvider"])
        ctrl = np.zeros((1, 12), np.float32)
        ctrl[0, 0], ctrl[0, 2] = 2.5, -1.5
        cmask = np.zeros((1, 12), np.float32)
        cmask[0, 0], cmask[0, 2] = 1, 1
        out = session.run(
            None,
            {
                "ids": np.array([[0, 1]], np.int64),
                "n": np.array([2], np.int64),
                "spk": np.array([[0.5, 0.25, 0, 0, 0, 0, 0, 0]], np.float32),
                "tag": np.array([0], np.int64),
                "ctrl": ctrl,
                "cmask": cmask,
            },
        )[0]
        print("echo output", out.shape, "ctrl", out[0, 0], out[0, 2],
              "mask", out[0, 12], "n", out[0, 13], "tag", out[0, 14], "spk", out[0, 15])


if __name__ == "__main__":
    main()
