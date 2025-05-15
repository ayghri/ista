import numpy as np


def kth_largest(arr, k, axis=None):
    """
    Finds the k-th largest value in an array, either along a specified axis
    or in the flattened array.

    Args:
      arr: NumPy array (can be multi-dimensional).
      k: The rank (1 for largest, 2 for second largest, etc.). Must be
         a positive integer.
      axis: The axis along which to find the k-th largest value.
            - If an integer, finds the k-th largest along that axis.
            - If None (default), finds the k-th largest in the entire
              flattened array.

    Returns:
      - If axis is an integer: NumPy array containing the k-th largest
        value along the specified axis. The shape of the returned array
        will be the same as the input array, but with the specified axis
        removed.
      - If axis is None: A scalar (0-D NumPy array) containing the single
        k-th largest value from the entire array.

    Raises:
      ValueError: If k is not a positive integer or is larger than the
                  relevant size (size of the specified axis or total size
                  if axis is None).
      ValueError: If the specified axis value (when not None) is invalid
                  (out of bounds) for the input array's dimensions or if
                  axis is not an integer or None.
      ValueError: If k is requested from an empty array or an empty axis.
    """
    arr = np.asarray(arr)

    # Validate k type and positivity (common check)
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    if axis is None:
        # --- Handle flattened array case (axis is None) ---
        size_along_axis = arr.size
        if size_along_axis == 0:
            raise ValueError(
                f"Cannot find k={k} smallest element in an empty array."
            )

        if k > size_along_axis:
            raise ValueError(
                f"k ({k}) cannot be larger than the total number of elements "
                f"({size_along_axis}) when axis is None"
            )

        # np.partition requires a positive index for k-th smallest.
        # To find the k-th largest, we need the (N - k)-th smallest element
        # in the 0-indexed partition. N is the total size.
        kth_smallest_index = size_along_axis - k

        # Partition the flattened array. arr.ravel() creates a flattened view.
        # axis=None is implicit when partitioning a 1D array.
        partitioned_array = np.partition(arr.ravel(), kth_smallest_index)

        # Return the single k-th largest value (scalar)
        return partitioned_array[kth_smallest_index]

    else:
        try:
            size_along_axis = arr.shape[axis]
        except IndexError:
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {arr.ndim}"
            ) from None  # Suppress context for cleaner error
        except TypeError:
            raise ValueError(
                f"axis must be an integer or None, not {type(axis)}"
            ) from None  # Suppress context

        if size_along_axis == 0:
            raise ValueError(
                f"Cannot find k={k} smallest element along axis {axis} with size 0."
            )

        if k > size_along_axis:
            raise ValueError(
                f"k ({k}) cannot be larger than the size of axis {axis} "
                f"({size_along_axis})"
            )

        kth_smallest_index = size_along_axis - k
        partitioned_array = np.partition(arr, kth_smallest_index, axis=axis)
        result = np.take(partitioned_array, kth_smallest_index, axis=axis)
        return result


if __name__ == "__main__":
    # --- Example Usage ---
    # Example 1: 2D array, default axis=None (flattened)
    arr2d = np.array(
        [
            [9, 1, 5, 2],
            [3, 8, 2, 6],
            [7, 4, 0, 9],
            [6, 0, 3, 5],
        ]
    )
    # Flattened: [9, 1, 5, 2, 3, 8, 2, 6, 7, 4, 0, 9, 6, 0, 3, 5]
    # Sorted desc: [9, 9, 8, 7, 6, 6, 5, 5, 4, 3, 3, 2, 2, 1, 0, 0]

    print("Original 2D array:")
    print(arr2d)

    # Find the 1st largest (max) in the flattened array (axis=None)
    largest_flat = kth_largest(arr2d, 1)  # axis=None is default
    print(
        f"\n1st largest (max) in flattened array (axis=None): {largest_flat}"
    )  # Expected: 9

    # Find the 3rd largest in the flattened array (axis=None)
    third_largest_flat = kth_largest(arr2d, 3)
    print(
        f"3rd largest in flattened array (axis=None): {third_largest_flat}"
    )  # Expected: 8

    # Find the 16th largest (min) in the flattened array (axis=None)
    smallest_flat = kth_largest(arr2d, 16)
    print(
        f"16th largest (min) in flattened array (axis=None): {smallest_flat}"
    )  # Expected: 0

    # Example 2: Using specified axis (same as before)
    # Find the 1st largest (max) in each column (axis=0)
    largest_col = kth_largest(arr2d, 1, axis=0)
    print(
        "\n1st largest (max) column-wise (axis=0):", largest_col
    )  # Expected: [9 8 5 9]

    # Find the 2nd largest in each row (axis=1)
    second_largest_row = kth_largest(arr2d, 2, axis=1)
    print(
        "2nd largest row-wise (axis=1):", second_largest_row
    )  # Expected: [5 6 7 5]
