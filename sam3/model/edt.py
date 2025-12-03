# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Triton kernel for euclidean distance transform (EDT)

On platforms where Triton is not available (e.g. Windows),
we provide a stub edt_triton() that simply raises a RuntimeError.

This keeps imports working for image-only use cases that never
actually call edt_triton (e.g. SAM3 image model), while still
supporting the real kernel on Linux + Triton.
"""

import torch

# Try to import Triton. This will fail on Windows.
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    """
    Disclaimer: This implementation is not meant to be extremely efficient. A CUDA kernel would likely be more efficient.
    Even in Triton, there may be more suitable algorithms.

    The goal of this kernel is to mimic cv2.distanceTransform(input, cv2.DIST_L2, 0).
    Recall that the euclidean distance transform (EDT) calculates the L2 distance to the closest zero pixel for each pixel of the source image.
    (original Meta docstring kept intact)
    """

    @triton.jit
    def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        # This is a somewhat verbatim implementation of the efficient 1D EDT algorithm described above
        # It can be applied horizontally or vertically depending if we're doing the first or second stage.
        # It's parallelized across batch+row (or batch+col if horizontal=False)
        # TODO: perhaps the implementation can be revisited if/when local gather/scatter become available in triton
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        # This will be the index of the right most parabola in the envelope ("the top of the stack")
        k = 0
        for q in range(1, length):
            # Read the function value at the current location. Note that we're doing a singular read, not very efficient
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            # location of the parabola on top of the stack
            r = tl.load(v + block_start + (k * stride))
            # associated boundary
            z_k = tl.load(z + block_start + (k * stride))
            # value of the function at the parabola location
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            # intersection between the two parabolas
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            # we'll pop as many parabolas as required
            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            # Store the new one
            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)

        # Last step, we read the envelope to find the min in every location
        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride), mask=(k + 1) < length, other=q
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)

    def edt_triton(data: torch.Tensor):
        """
        Computes the Euclidean Distance Transform (EDT) of a batch of binary images.

        Args:
            data: A tensor of shape (B, H, W) representing a batch of binary images.

        Returns:
            A tensor of the same shape as data containing the EDT.
            It should be equivalent to a batched version of
            cv2.distanceTransform(input, cv2.DIST_L2, 0)
        """
        assert data.dim() == 3
        assert data.is_cuda
        B, H, W = data.shape
        data = data.contiguous()

        # Allocate the "function" tensor. Implicitly the function is
        # 0 if data[i,j]==0 else +infinity
        output = torch.where(data, 1e18, 0.0)
        assert output.is_contiguous()

        # Scratch tensors for the parabola stacks
        parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
        parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18

        # Grid size (number of blocks)
        grid = (B, H)

        # Launch horizontal pass
        edt_kernel[grid](
            output.clone(),
            output,
            parabola_loc,
            parabola_inter,
            H,
            W,
            horizontal=True,
        )

        # reset the parabola stacks
        parabola_loc.zero_
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18

        # Vertical pass
        grid = (B, W)
        edt_kernel[grid](
            output.clone(),
            output,
            parabola_loc,
            parabola_inter,
            H,
            W,
            horizontal=False,
        )
        # don't forget to take sqrt at the end
        return output.sqrt()

else:
    # -------------------------------------------------------------------------
    # Stub for platforms without Triton (e.g. Windows)
    # -------------------------------------------------------------------------
    def edt_triton(data: torch.Tensor):
        """
        Stub EDT implementation for environments where Triton is not available.

        This function is only used by certain tracking/video paths.
        The SAM3 *image-only* APIs do not require it.

        On Windows, this stub allows `from sam3.model.edt import edt_triton`
        to succeed, while making it clear that the Triton kernel itself
        cannot run here.
        """
        raise RuntimeError(
            "edt_triton() was called but Triton is not available on this platform.\n"
            "This EDT kernel is only supported on Linux with Triton installed.\n"
            "If you're only using the SAM3 image model (no video/tracking), "
            "nothing should call edt_triton(), and you can safely ignore this.\n"
        )
