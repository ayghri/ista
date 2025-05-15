import torch
from typing import Optional


@torch.no_grad()
def hard_threshold_inplace(
    tensor: torch.Tensor, k: int, dim: Optional[int] = None
) -> torch.Tensor:
    """
    Performs inplace hard thresholding on a tensor.

    This function keeps the `k` elements with the largest absolute values
    and sets all other elements to zero. The operation is performed inplace.

    Args:
        tensor (torch.Tensor): The input tensor to be thresholded.
                               This tensor will be modified directly.
        k (int): The number of elements to keep.
                 If `k <= 0`, all elements in the tensor (or along the specified
                 dimension) are set to zero.
                 If `k` is greater than or equal to the number of elements
                 (globally or along `dim`), the tensor remains unchanged.
        dim (Optional[int], optional): The dimension along which to apply
                                       the hard thresholding.
                                       If `None` (default), thresholding is
                                       applied globally across all elements
                                       of the flattened tensor.
                                       If an integer, thresholding is applied
                                       independently for each slice along
                                       the specified dimension.

    Returns:
        torch.Tensor: The input `tensor` after inplace hard thresholding.
    """
    if k < 0:
        return tensor
    if k == 0:
        tensor.zero_()
        return tensor

    if dim is None:
        num_elements = tensor.numel()
        if k >= num_elements:
            return tensor

        flat_tensor_view = tensor.view(-1)
        abs_values_flat = torch.abs(flat_tensor_view)

        _, top_k_indices_flat = torch.topk(
            abs_values_flat, k, largest=True, sorted=False
        )

        mask_flat = torch.zeros_like(abs_values_flat, dtype=torch.bool)
        mask_flat[top_k_indices_flat] = True

        flat_tensor_view.mul_(mask_flat)

    else:
        if not (0 <= dim < tensor.ndim):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [0, {tensor.ndim-1}], but got {dim})"
            )

        dim_size = tensor.shape[dim]
        if k >= dim_size:
            return tensor

        abs_values_dim = torch.abs(tensor)

        _, top_k_indices_dim = torch.topk(
            abs_values_dim, k, dim=dim, largest=True, sorted=False
        )

        mask_dim = torch.zeros_like(tensor, dtype=torch.bool)
        mask_dim.scatter_(dim, top_k_indices_dim, True)

        tensor.mul_(mask_dim)

    return tensor


@torch.no_grad()
def soft_threshold_inplace(
    x: torch.Tensor, thresh: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    """
    Inplace soft-thresholding of a tensor `x`.

    The soft-thresholding operation is defined as:
    S_lambda(x_i) = sign(x_i) * max(0, |x_i| - lambda_i)

    The threshold `thresh` can be applied globally, element-wise, or along a specific dimension.
    Threshold values from `thresh` are internally clamped to be non-negative.

    Args:
        x (torch.Tensor): Input tensor to be thresholded inplace. This tensor will be modified.
        thresh (torch.Tensor): Tensor of threshold values. Its interpretation depends on `dim`:
            - If `dim` is None (default):
                - `thresh` can be a scalar tensor (e.g., `torch.tensor(0.5)`):
                  The same threshold is applied to all elements of `x`.
                - `thresh` can be a tensor of the same shape as `x`:
                  Thresholds are applied element-wise.
            - If `dim` is an int (specifies a dimension of `x`):
                - `thresh` can be a scalar tensor:
                  The same threshold is applied to all slices along the `dim`-th dimension of `x`.
                - `thresh` can be a 1D tensor: Its length must match `x.shape[dim]`.
                  Each element of `thresh` is applied to the corresponding slice along `dim`.
            Note: Values in `thresh` will be clamped to be non-negative before use.
        dim (Optional[int], optional): The dimension along which to apply thresholds.
            If None, `thresh` is applied globally (if scalar) or element-wise (if same shape as `x`).
            If an int, `thresh` is applied along this dimension of `x`. Defaults to None.

    Returns:
        torch.Tensor: The input `x` tensor, modified inplace.

    Raises:
        ValueError: If `thresh` and `dim` combinations are incompatible, `dim` is out of range,
                    or `thresh` has an unsupported shape for the given `dim` configuration.
    """
    _thresh_internal = torch.clamp_min(
        thresh.to(dtype=x.dtype, device=x.device), 0.0
    )

    if dim is not None:
        if not (isinstance(dim, int) and 0 <= dim < x.ndim):
            raise ValueError(
                f"dim {dim} out of range for tensor with {x.ndim} dimensions. "
                f"Must be an integer between 0 and {x.ndim - 1}."
            )

        current_thresh_shape = _thresh_internal.shape
        current_thresh_ndim = _thresh_internal.ndim
        current_thresh_numel = _thresh_internal.numel()

        if current_thresh_numel == 1:
            scalar_as_1d = _thresh_internal.reshape(1)  # Makes it shape (1,)
            _effective_1d_thresh = scalar_as_1d.expand(x.shape[dim])
        elif (
            current_thresh_ndim == 1 and current_thresh_shape[0] == x.shape[dim]
        ):
            # thresh is already a 1D tensor of the correct size for the specified dimension
            _effective_1d_thresh = _thresh_internal
        else:
            # thresh is not a scalar and not a compatible 1D tensor
            expected_len_str = str(x.shape[dim])
            actual_len_str = (
                str(current_thresh_shape[0])
                if current_thresh_ndim == 1
                else "N/A (not 1D)"
            )
            raise ValueError(
                f"When dim ({dim}) is specified, thresh must be a scalar tensor or a 1D tensor "
                f"whose length ({actual_len_str}) matches x.shape[dim] ({expected_len_str}). "
                f"Got prepared thresh with shape: {current_thresh_shape}."
            )

        # Reshape _effective_1d_thresh for broadcasting with x
        view_shape = [1] * x.ndim
        view_shape[dim] = x.shape[dim]
        thresh_for_op = _effective_1d_thresh.view(view_shape)

    else:  # dim is None (Global or element-wise thresholding)
        current_thresh_numel = _thresh_internal.numel()
        if current_thresh_numel == 1:  # Scalar threshold for global application
            thresh_for_op = _thresh_internal
        elif _thresh_internal.shape == x.shape:  # Element-wise thresholds
            thresh_for_op = _thresh_internal
        else:
            raise ValueError(
                "When dim is None, thresh must be a scalar tensor or a tensor with the same shape as x. "
                f"Got x.shape={x.shape} and prepared thresh.shape={_thresh_internal.shape}."
            )

    if torch.all(thresh_for_op == 0):
        return x

    signs = torch.sign(x)
    x.abs_()  # x = |x|
    x.sub_(thresh_for_op)  # x = |x| - thresh_for_op (broadcasting applies here)
    x.relu_()  # x = max(0, |x| - thresh_for_op)
    x.mul_(signs)  # x = sign(x) * max(0, |x| - thresh_for_op)

    return x
