from typing import Callable, Union
import torch

class PlaceholderAsScalar:
    def __init__(self, scalar_type):
        """
        This class is used to decorate a parameter that is a placeholder for a scalar value. It is untested to
        use this to annotate parameters of functions being traced with the purpose of actual code generation.
        The main use case is to track scalar placeholders within fx graphs in order to obtain fx graphs
        that are suitable as Patterns for graph matching / replacement, evscalar_typeen in the presence of scalar placeholders
        or intermediate nodes.scalar_type

        :param scalar_type: Either float, int, or bool
        """
        if scalar_type not in (float, int, bool):
            raise ValueError("scalar_type must be one of float, int, or bool")
        self.scalar_type = scalar_type
    def example_value(self, original_example_value : torch.Tensor):
        """
        Convert a Tensor example value to a scalar example value using the
        provided scalar_type
        """
        assert original_example_value.numel() == 1, "PlaceholderAsScalar can only be used on tensors with a single element"
        return self.scalar_type(original_example_value)
