import torch
import time
import numpy as np
from thop import profile


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


def get_frames_per_second(
    model,
    device,
    sequence_size,
    movinet,
    frame_size=224,
    iterations=10,
    init_iterations=2,
    batch_size=1,
    use_jit=False,
    use_half=False,
):

    dtype = torch.half if use_half else torch.float32
    model = model.eval().requires_grad_(False).to(dtype).to(device)
    frames = torch.randn(
        (batch_size * iterations, 3, sequence_size, frame_size, frame_size)
    ).to(dtype)
    if movinet:
        frames = frames.permute(0, 1, 2, 3, 4)  # [B, C, S, H, W]
        print(frames.shape)
    print("Test with data")

    times = []
    for i in range(iterations):
        print(f"Iteration: {i}")
        for batch_index, frames_batch in enumerate(batch(frames, n=batch_size)):
            time1 = time.time()
            cuda_batch = frames_batch.to(device)
            _ = model(cuda_batch).cpu()
            time2 = time.time()
            iter_time = time2 - time1
            times.append([iter_time])

    total_iterations = len(times)
    print(f"Total batches: {total_iterations}")
    print(f"Batch Size: {batch_size} | Use Half: {use_half} | Use JIT: {use_jit} | ")
    fps = 1 / (
        np.sum(times[init_iterations:])
        / (total_iterations - init_iterations)
        / batch_size
    )
    print(f"Total speed: {fps} FPS")


def get_flops(model, seq_len, movinet=False, write_path=None):
    input = torch.rand(1, seq_len, 3, 224, 224)
    if movinet:
        input = input.permute(0, 2, 1, 3, 4)
    macs, params = profile(model, inputs=(input,))
    flops = 2 * macs
    print(f"# params: {params} | MACs: {macs} | FLOPs: {flops}")
