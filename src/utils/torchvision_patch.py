"""
Patch for torchvision compatibility with basicsr package.
This module provides compatibility between newer torchvision versions and basicsr.
"""

import sys
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader


class TorchvisionCompatLoader(Loader):
    """Compatibility loader for torchvision transforms."""
    
    def exec_module(self, _):
        """Execute the module, redirecting old imports to new locations."""
        from torchvision.transforms import _functional_tensor
        sys.modules['torchvision.transforms.functional_tensor'] = _functional_tensor


class TorchvisionCompatFinder(MetaPathFinder):
    """Finder for torchvision compatibility modules."""
    
    def find_spec(self, fullname, _, __):
        """Find and return the module specification if it's the target module."""
        if fullname == 'torchvision.transforms.functional_tensor':
            return spec_from_loader(
                fullname,
                TorchvisionCompatLoader(),
                is_package=False
            )
        return None


def apply_patch():
    """Apply the torchvision compatibility patch."""
    sys.meta_path.insert(0, TorchvisionCompatFinder()) 
