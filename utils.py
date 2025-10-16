"""
Utility classes and functions for LED Solar Simulator
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class LEDSpec:
    """LED specification data structure"""
    name: str
    peak_wavelength: float
    viewing_angle: float
    wavelengths: np.ndarray
    intensities: np.ndarray
    manufacturer: Optional[str] = None
    part_number: Optional[str] = None
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays"""
        if not isinstance(self.wavelengths, np.ndarray):
            self.wavelengths = np.array(self.wavelengths)
        if not isinstance(self.intensities, np.ndarray):
            self.intensities = np.array(self.intensities)