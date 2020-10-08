import os
import torch


def mkdir(dir):
    try:
        os.makedirs(dir)
        print('Created directory', dir)
    except OSError:
        pass


def embed_tensor_complex(sample):
    s = sample.unsqueeze(-1)
    zeros = s.new_zeros(size=s.size())
    return torch.cat([s, zeros], axis=-1)


"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

def complex_mul(x, y):
    """
    Complex multiplication.
    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.
    Args:
        x (torch.Tensor): A PyTorch tensor with the last dimension of size 2.
        y (torch.Tensor): A PyTorch tensor with the last dimension of size 2.
    Returns:
        torch.Tensor: A PyTorch tensor with the last dimension of size 2.
    """
    assert x.shape[-1] == y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x):
    """
    Complex conjugate.
    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.
    Args:
        x (torch.Tensor): A PyTorch tensor with the last dimension of size 2.
        y (torch.Tensor): A PyTorch tensor with the last dimension of size 2.
    Returns:
        torch.Tensor: A PyTorch tensor with the last dimension of size 2.
    """
    assert x.shape[-1] == 2

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def fft2c(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3
            dimensions: dimensions -3 & -2 are spatial dimensions and dimension
            -1 has size 2. All other dimensions are assumed to be batch
            dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    # data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))

    return data


def ifft2c(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3
            dimensions: dimensions -3 & -2 are spatial dimensions and dimension
            -1 has size 2. All other dimensions are assumed to be batch
            dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    # data = fftshift(data, dim=(-3, -2))

    return data


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the
            final dimension should be 2.
    Returns:
        torch.Tensor: Absolute value of data.
    """
    assert data.size(-1) == 2

    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_sq(data):
    """
    Compute the squared absolute value of a complex tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the
            final dimension should be 2.
    Returns:
        torch.Tensor: Squared absolute value of data.
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1)


# Helper functions


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors.
    Args:
        x (torch.Tensor): A PyTorch tensor.
        shift (int): Amount to roll.
        dim (int): Which dimension to roll.
    Returns:
        torch.Tensor: Rolled version of x.
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    Args:
        x (torch.Tensor): A PyTorch tensor.
        dim (int): Which dimension to fftshift.
    Returns:
        torch.Tensor: fftshifted version of x.
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]

    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    Args:
        x (torch.Tensor): A PyTorch tensor.
        dim (int): Which dimension to ifftshift.
    Returns:
        torch.Tensor: ifftshifted version of x.
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]

    return roll(x, shift, dim)


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.
    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]



#### Copyright @Zak

from odl import Operator
class OperatorFunction(torch.autograd.Function):

    """Wrapper of an ODL operator as a ``torch.autograd.Function``.
    """

    @staticmethod
    def forward(ctx, operator, input):
        """Evaluate forward pass on the input.
        Parameters
        ----------
        ctx : context object
            Object to communicate information between forward and backward
            passes.
        operator : `Operator`
            ODL operator to be wrapped. For gradient computations to
            work, ``operator.derivative(x).adjoint`` must be implemented.
        input : `torch.Tensor`
            Point at which to evaluate the operator.
        Returns
        -------
        result : `torch.Tensor`
            Tensor holding the result of the evaluation.
        """
        if not isinstance(operator, Operator):
            raise TypeError(
                "`operator` must be an `Operator` instance, got {!r}"
                "".format(operator)
            )

        # Save operator for backward; input only needs to be saved if
        # the operator is nonlinear (for `operator.derivative(input)`)
        ctx.operator = operator

        if not operator.is_linear:
            # Only needed for nonlinear operators
            ctx.save_for_backward(input)

        # TODO(kohr-h): use GPU memory directly when possible
        # TODO(kohr-h): remove `copy_if_zero_strides` when NumPy 1.16.0
        # is required
        input_arr = copy_if_zero_strides(input.cpu().detach().numpy())

        # Determine how to loop over extra shape "left" of the operator
        # domain shape
        in_shape = input_arr.shape
        op_in_shape = operator.domain.shape
        if operator.is_functional:
            op_out_shape = ()
            op_out_dtype = operator.domain.dtype
        else:
            op_out_shape = operator.range.shape
            op_out_dtype = operator.range.dtype

        extra_shape = in_shape[:-len(op_in_shape)]
        if in_shape[-len(op_in_shape):] != op_in_shape:
            shp_str = str(op_in_shape).strip('(,)')
            raise ValueError(
                'input tensor has wrong shape: expected (*, {}), got {}'
                ''.format(shp_str, in_shape)
            )

        # Store some information on the context object
        ctx.op_in_shape = op_in_shape
        ctx.op_out_shape = op_out_shape
        ctx.extra_shape = extra_shape
        ctx.op_in_dtype = operator.domain.dtype
        ctx.op_out_dtype = op_out_dtype

        # Evaluate the operator on all inputs in a loop
        if extra_shape:
            # Multiple inputs: flatten extra axes, then do one entry at a time
            input_arr_flat_extra = input_arr.reshape((-1,) + op_in_shape)
            results = []
            for inp in input_arr_flat_extra:
                results.append(operator(inp))

            # Stack results, reshape to the expected output shape and enforce
            # correct dtype
            result_arr = np.stack(results).astype(op_out_dtype, copy=False)
            result_arr = result_arr.reshape(extra_shape + op_out_shape)
        else:
            # Single input: evaluate directly
            result_arr = np.asarray(
                operator(input_arr)
            ).astype(op_out_dtype, copy=False)

        # Convert back to tensor
        tensor = torch.from_numpy(result_arr).to(input.device)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):

        # Return early if there's nothing to do
        if not ctx.needs_input_grad[1]:
            return None, None

        operator = ctx.operator

        # Get `operator` and `input` from the context object (the input
        # is only needed for nonlinear operators)
        if not operator.is_linear:
            # TODO: implement directly for GPU data
            # TODO(kohr-h): remove `copy_if_zero_strides` when NumPy 1.16.0
            # is required
            input_arr = copy_if_zero_strides(
                ctx.saved_tensors[0].detach().cpu().numpy()
            )

        # ODL weights spaces, pytorch doesn't, so we need to handle this
        try:
            dom_weight = operator.domain.weighting.const
        except AttributeError:
            dom_weight = 1.0
        try:
            ran_weight = operator.range.weighting.const
        except AttributeError:
            ran_weight = 1.0
        scaling = dom_weight / ran_weight

        # Convert `grad_output` to NumPy array
        grad_output_arr = copy_if_zero_strides(
            grad_output.detach().cpu().numpy()
        )

        # Get shape information from the context object
        op_in_shape = ctx.op_in_shape
        op_out_shape = ctx.op_out_shape
        extra_shape = ctx.extra_shape
        op_in_dtype = ctx.op_in_dtype

        # Check if `grad_output` is consistent with `extra_shape` and
        # `op_out_shape`
        if grad_output_arr.shape != extra_shape + op_out_shape:
            raise ValueError(
                'expected tensor of shape {}, got shape {}'
                ''.format(extra_shape + op_out_shape, grad_output_arr.shape)
            )

        # Evaluate the (derivative) adjoint on all inputs in a loop
        if extra_shape:
            # Multiple gradients: flatten extra axes, then do one entry
            # at a time
            grad_output_arr_flat_extra = grad_output_arr.reshape(
                (-1,) + op_out_shape
            )

            results = []
            if operator.is_linear:
                for ograd in grad_output_arr_flat_extra:
                    results.append(np.asarray(operator.adjoint(ograd)))
            else:
                # Need inputs, flattened in the same way as the gradients
                input_arr_flat_extra = input_arr.reshape((-1,) + op_in_shape)
                for ograd, inp in zip(
                    grad_output_arr_flat_extra, input_arr_flat_extra
                ):
                    results.append(
                        np.asarray(operator.derivative(inp).adjoint(ograd))
                    )

            # Stack results, reshape to the expected output shape and enforce
            # correct dtype
            result_arr = np.stack(results).astype(op_in_dtype, copy=False)
            result_arr = result_arr.reshape(extra_shape + op_in_shape)
        else:
            # Single gradient: evaluate directly
            if operator.is_linear:
                result_arr = np.asarray(
                    operator.adjoint(grad_output_arr)
                ).astype(op_in_dtype, copy=False)
            else:
                result_arr = np.asarray(
                    operator.derivative(input_arr).adjoint(grad_output_arr)
                ).astype(op_in_dtype, copy=False)

        # Apply scaling, convert to tensor and return
        if scaling != 1.0:
            result_arr *= scaling
        grad_input = torch.from_numpy(result_arr).to(grad_output.device)