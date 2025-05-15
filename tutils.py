import torch
import numpy as np  # For comparison data generation
from torch import nn
import math


def kth_largest_torch(tensor: torch.Tensor, k: int, dim=None):
    """
    Finds the k-th largest value in a tensor, either along a specified dimension
    or in the flattened tensor. Uses torch.kthvalue.

    Args:
      tensor_in: Input PyTorch Tensor (can be multi-dimensional).
      k: The rank (1 for largest, 2 for second largest, etc.). Must be
         a positive integer.
      dim: The dimension along which to find the k-th largest value.
            - If an integer, finds the k-th largest along that dimension.
            - If None (default), finds the k-th largest in the entire
              flattened tensor.

    Returns:
      - If dim is an integer: Tensor containing the k-th largest
        value along the specified dimension. The shape of the returned tensor
        will be the same as the input tensor, but with the specified dimension
        removed (unless keepdim=True, which is not the default here).
      - If dim is None: A scalar (0-D Tensor) containing the single
        k-th largest value from the entire tensor.

    Raises:
      ValueError: If k is not a positive integer or is larger than the
                  relevant size (size of the specified dimension or total size
                  if dim is None).
      ValueError: If the specified dim value (when not None) is invalid
                  (out of bounds) for the input tensor's dimensions or if
                  dim is not an integer or None.
      ValueError: If k is requested from an empty tensor or an empty dimension.
      RuntimeError: Potentially from internal PyTorch operations if inputs
                    are inconsistent (e.g., k=0).
    """

    # --- Common Validation ---
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    # --- Handle Flattened Case (dim is None) ---
    if dim is None:
        numel = tensor.numel()
        if numel == 0:
            raise ValueError(
                f"Cannot find k={k} largest element in an empty tensor."
            )

        if k > numel:
            raise ValueError(
                f"k ({k}) cannot be larger than the total number of elements "
                f"({numel}) when dim is None"
            )

        # torch.kthvalue finds the k-th *smallest* value (1-based k).
        # To find the k-th *largest* (1-based k_user), we need the
        # (N - k_user + 1)-th smallest element, where N is the total size.
        k_torch = numel - k + 1

        # Flatten the tensor before finding the k-th value
        flat_tensor = tensor.flatten()

        # kthvalue returns a named tuple (values, indices)
        result = torch.kthvalue(flat_tensor, k_torch)
        return result.values  # Return only the value (0-D tensor)

    # --- Handle Specific Dimension Case (dim is an integer) ---
    else:
        # Validate dim type first
        if not isinstance(dim, int):
            raise ValueError(f"dim must be an integer or None, not {type(dim)}")

        # Validate dim bounds
        ndim = tensor.dim()
        if not -ndim <= dim < ndim:
            raise ValueError(
                f"Dimension {dim} is out of bounds for tensor of dimension {ndim}"
            )

        # Handle negative dimension indexing
        if dim < 0:
            dim = ndim + dim

        # Get size along the specified dimension
        size_along_dim = tensor.shape[dim]

        if size_along_dim == 0:
            raise ValueError(
                f"Cannot find k={k} largest element along dimension {dim} with size 0."
            )

        # Validate k against the size of the specified dimension
        if k > size_along_dim:
            raise ValueError(
                f"k ({k}) cannot be larger than the size of dimension {dim} "
                f"({size_along_dim})"
            )

        # As before, convert k-th largest to (N - k + 1)-th smallest for torch.kthvalue
        k_torch = size_along_dim - k + 1

        # Find the k-th smallest value along the specified dimension
        # keepdim=False removes the dimension
        result = torch.kthvalue(tensor, k_torch, dim=dim, keepdim=False)
        return result.values  # Return only the values tensor


# Helper function to calculate k-th largest absolute value safely
@torch.no_grad()
def calculate_kth_value(vec, k):
    if vec is None or vec.numel() == 0:
        return 0.0
    N = vec.numel()
    k = max(1, min(N, int(k)))  # Ensure 1 <= k <= N
    kth_smallest_index = max(1, N - k + 1)
    try:
        kth_largest_abs_val = torch.kthvalue(
            vec.abs().flatten(), kth_smallest_index
        ).values
    except RuntimeError as e:
        print(
            f"Warning: kthvalue failed (N={N}, k={k}, index={kth_smallest_index}). Error: {e}. Returning 0."
        )
        return 0.0
    return kth_largest_abs_val.item()


@torch.no_grad()
def calculate_sparsity_per_layer(model, threshold=1e-8):
    sparsity_dict = {}
    total_sparsity_num = 0
    total_sparsity_den = 0
    param_idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            # zero_params = (param.abs() < threshold).sum().item()
            zero_params = numel - (param.abs() > 0).sum().item()
            layer_sparsity = (zero_params / numel) * 100.0 if numel > 0 else 0.0
            key_name = f"L{param_idx}_{name}"
            status_tags = []
            if status_tags:
                key_name += f" [{', '.join(status_tags)}]"
            sparsity_dict[key_name] = layer_sparsity
            total_sparsity_num += zero_params
            total_sparsity_den += numel
            param_idx += 1
    overall_sparsity = (
        (total_sparsity_num / total_sparsity_den) * 100.0
        if total_sparsity_den > 0
        else 0.0
    )
    return overall_sparsity, sparsity_dict


@torch.no_grad()  # Ensure no gradients are computed within this function
def print_model_sparsity(model: nn.Module, threshold: float = 1e-8):
    """
    Calculates and prints the sparsity of a PyTorch model, including total parameters.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        threshold (float): Absolute value threshold below which a parameter
                           is considered zero. Defaults to 1e-8.
    """
    print("-" * 90)  # Adjusted width for new column
    print(f"Sparsity Analysis (Threshold: {threshold:.1e})")
    print("-" * 90)
    print(
        f"{'Parameter Name':<45} {'Shape':<20} {'NNZ':<12} {'Total':<12} {'Sparsity (%)':<15}"
    )
    print("=" * 90)

    total_params_overall = 0  # Includes non-trainable if you iterate over all
    total_params_trainable = 0
    total_nnz_trainable = 0

    for name, param in model.named_parameters():
        # Track all parameters for total count
        numel = param.numel()
        total_params_overall += numel  # Count all params encountered

        # Only consider trainable parameters for sparsity stats
        if param.requires_grad:
            if numel == 0:  # Skip empty parameters if any
                print(
                    f"{name:<45} {str(tuple(param.shape)):<20} {'N/A':<12} {numel:<12,d} {'N/A':<15}"
                )
                continue

            # Calculate non-zero elements based on the threshold
            if hasattr(param, "dystil_k"):
                nnz = (param.abs() > threshold).sum().item()
            else:
                nnz = param.numel()

            sparsity = (1.0 - nnz / numel) * 100.0

            total_params_trainable += numel
            total_nnz_trainable += nnz

            print(
                f"{name:<45} {str(tuple(param.shape)):<20} {nnz:<12,d} {numel:<12,d} {sparsity:<15.2f}"
            )
        else:
            # Print info for non-trainable parameters
            print(
                f"{name:<45} {str(tuple(param.shape)):<20} {'N/A (NT)':<12} {numel:<12,d} {'N/A (NT)':<15}"
            )

    print("-" * 90)
    if total_params_trainable > 0:
        overall_sparsity = (
            1.0 - total_nnz_trainable / total_params_trainable
        ) * 100.0
        print(
            f"{'Overall Trainable':<45} {' ':<20} {total_nnz_trainable:<12,d} {total_params_trainable:<12,d} {overall_sparsity:<15.2f}"
        )
        # print(f"(Total Trainable Parameters: {total_params_trainable:,d})") # Included in summary line
    else:
        print("Model has no trainable parameters.")
    # Optionally print total overall parameters if different from trainable
    # if total_params_overall != total_params_trainable:
    #      print(f"(Total Overall Parameters: {total_params_overall:,d})")
    print("-" * 90)


@torch.no_grad()
def initialize_sparse(
    model: nn.Module,
    # weight_init_fn=nn.init.kaiming_normal_,
    init_kwargs=None,
):
    """
    Initializes model parameters, applying target sparsity to specified weights.

    Args:
        model (nn.Module): The model to initialize.
        weight_init_fn (callable, optional): Initialization function for weights
            (e.g., nn.init.kaiming_normal_, nn.init.xavier_normal_).
            Defaults to kaiming_normal_.
        bias_init_fn (callable, optional): Initialization function for biases.
            Defaults to zeros_.
        init_kwargs (dict, optional): Additional keyword arguments to pass to the
            weight_init_fn (e.g., {'nonlinearity': 'relu'} for Kaiming).
    """
    if init_kwargs is None:
        init_kwargs = {}

    for name, p in model.named_parameters():
        if not p.requires_grad or not hasattr(p, "dystil_k"):
            # print(f"  Skipping {name} (requires_grad=False)")
            continue

        # try:
        # weight_init_fn(p, **init_kwargs)
        # except TypeError:
        # weight_init_fn(p)

        # Apply sparsity if target is less than 100%
        numel = p.numel()
        if numel == 0:
            continue  # Skip empty tensors

        # Calculate k: number of elements to keep non-zero
        k = p.dystil_k
        if k < numel:  # Only prune if k is less than total elements
            abs_p = p.abs().flatten()
            # Find the threshold: k-th largest absolute value is (N-k+1)-th smallest
            try:
                # Use max(1,...) defensively for index
                threshold_index = max(1, numel - k + 1)
                threshold_val = torch.kthvalue(abs_p, threshold_index).values
                # Create mask where abs(value) >= threshold
                mask = p.abs() >= threshold_val
                # Apply mask: set elements below threshold to 0
                p.data.mul_(mask)
                # print(f"    Sparsified {name}: kept {mask.sum().item()}/{numel} elements (target k={k})")
            except Exception as e:
                print(
                    f"    Warning: Could not sparsify {name} (numel={numel}, k={k}). Error: {e}"
                )
            print(
                name,
                p.numel(),
                p.shape,
                p.dystil_k,
                p.dystil_beta,
                "NNZ:",
                mask.sum().item(),
            )

    print("Sparse initialization complete.")


# Helper function for soft-thresholding
@torch.no_grad()
def soft_threshold(x, thresh):
    thresh = torch.tensor(max(0.0, thresh), dtype=x.dtype, device=x.device)
    return torch.sign(x) * torch.relu(torch.abs(x) - thresh)



if __name__ == "__main__":
    # --- Example Usage (remains the same, just using 'dim' argument name) ---

    # Example 1: 2D tensor, default dim=None (flattened)
    arr_np = np.array([[9, 1, 5, 2], [3, 8, 2, 6], [7, 4, 0, 9], [6, 0, 3, 5]])
    tensor2d = torch.from_numpy(arr_np)

    print("Original 2D Tensor:")
    print(tensor2d)

    # Find the 1st largest (max) in the flattened tensor (dim=None)
    largest_flat = kth_largest_torch(tensor2d, 1)  # dim=None is default
    print(f"\n1st largest (max) in flattened tensor (dim=None): {largest_flat}")

    # Find the 3rd largest in the flattened tensor (dim=None)
    third_largest_flat = kth_largest_torch(tensor2d, 3)
    print(f"3rd largest in flattened tensor (dim=None): {third_largest_flat}")

    # Example 2: Using specified dimension
    # Find the 1st largest (max) in each column (dim=0)
    largest_col = kth_largest_torch(tensor2d, 1, dim=0)
    print("\n1st largest (max) column-wise (dim=0):", largest_col)

    # Find the 2nd largest in each row (dim=1)
    second_largest_row = kth_largest_torch(tensor2d, 2, dim=1)
    print("2nd largest row-wise (dim=1):", second_largest_row)

    # Example 3: 3D tensor
    arr3d_np = np.arange(24).reshape(2, 3, 4)
    tensor3d = torch.from_numpy(arr3d_np)
    print("\nOriginal 3D Tensor:")
    print(tensor3d)

    # Find the 2nd largest along dim=1
    second_largest_ax1 = kth_largest_torch(tensor3d, 2, dim=1)
    print(f"\n2nd largest along dim=1:")
    print(second_largest_ax1)
