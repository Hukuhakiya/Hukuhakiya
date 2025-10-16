"""
LED Solar Simulator Optimizer - Enhanced Version with Advanced EA Improvements
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import lsq_linear
from typing import List, Tuple, Dict, Optional
import json
import os
import pandas as pd
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
from collections import defaultdict
import itertools
warnings.filterwarnings('ignore')

from utils import LEDSpec


class LEDCombination:
    """Represents a combination of LEDs with power settings"""
    def __init__(self, led_indices: List[int], powers: List[float], 
                 fitness: float = float('inf'), n_target: int = None):
        self.led_indices = led_indices
        self.powers = powers
        self.fitness = fitness
        self.n_target = n_target


# ============================================================================
# ENHANCED MODULE-LEVEL FUNCTIONS
# ============================================================================

def fit_powers_nnls_enhanced(led_indices, led_spectra, target_spectrum, 
                             wavelengths, weights, led_efficiencies, pmax, 
                             led_database, min_power_threshold=0.05, max_iterations=3):
    """Enhanced power fitting with iterative refinement"""
    if not led_indices:
        return []
    
    current_indices = led_indices.copy()
    
    for iteration in range(max_iterations):
        A = np.stack([led_spectra[i] for i in current_indices], axis=1)
        b = target_spectrum
        
        W = np.sqrt(weights + 1e-12)[:, None]
        Aw = W * A
        bw = W[:, 0] * b
        
        target_total = np.trapz(target_spectrum, wavelengths)
        max_bounds = []
        
        for idx in current_indices:
            peak = led_database[idx].peak_wavelength
            efficiency = led_efficiencies[idx]
            
            # Region-specific power limits
            if 600 <= peak <= 800:
                power_multiplier = 2.0  # Critical region
            elif 400 <= peak <= 600 or 800 <= peak <= 1000:
                power_multiplier = 1.5
            elif peak < 300 or peak > 1200:
                power_multiplier = 0.02
            elif peak < 350 or peak > 1150:
                power_multiplier = 0.15
            else:
                power_multiplier = 1.0
            
            if efficiency > 1e-12:
                max_power = (0.4 * target_total / efficiency) * power_multiplier
            else:
                max_power = pmax * power_multiplier
            
            max_bounds.append(min(pmax * power_multiplier, max_power))
        
        res = lsq_linear(Aw, bw, bounds=(0.0, max_bounds), max_iter=1000, verbose=0)
        powers = res.x
        
        # Remove LEDs with power below threshold
        if iteration < max_iterations - 1:  # Don't remove on last iteration
            low_power_mask = powers < min_power_threshold
            if np.any(low_power_mask) and len(current_indices) > 3:
                # Remove lowest power LED
                min_power_idx = np.argmin(powers)
                current_indices.pop(min_power_idx)
                powers = np.delete(powers, min_power_idx)
            else:
                break
        else:
            break
    
    return powers.tolist()


def calculate_fitness_enhanced(led_indices, powers, led_spectra, target_spectrum,
                               wavelengths, weights, n_target, dynamic_weights=None):
    """Enhanced fitness calculation with dynamic weights"""
    # Combined spectrum
    combined = np.zeros_like(target_spectrum)
    for led_idx, power in zip(led_indices, powers):
        combined += power * led_spectra[led_idx]
    
    # Use dynamic weights if provided, otherwise use static weights
    current_weights = dynamic_weights if dynamic_weights is not None else weights
    
    # RLS calculation
    denominator = target_spectrum + 1e-12
    numerator = (target_spectrum - combined) ** 2
    rls_values = numerator / denominator
    weighted_rls = rls_values * current_weights
    RLS = np.sum(weighted_rls) / np.sum(current_weights)
    
    # CL* calculation
    bins = [
        (300, 400), (400, 500), (500, 600), (600, 700), (700, 800),
        (800, 900), (900, 1000), (1000, 1100), (1100, 1200)
    ]
    
    CLs = []
    for lambda1, lambda2 in bins:
        mask = (wavelengths >= lambda1) & (wavelengths <= lambda2)
        numerator = np.trapz(combined[mask], wavelengths[mask])
        denominator = np.trapz(target_spectrum[mask], wavelengths[mask])
        SR = numerator / denominator if denominator > 1e-12 else 0.0
        CL = abs(1.0 - SR)
        CLs.append(CL)
    
    CL_star = max(CLs)
    nLedsOn = len(led_indices)
    RLSCL = RLS + CL_star
    
    # FF4: Piecewise fitness
    if CL_star <= 0.25 and nLedsOn <= n_target:
        fitness = RLSCL
    elif CL_star <= 0.25 and nLedsOn > n_target:
        fitness = nLedsOn * (RLSCL + 1)
    elif CL_star > 0.25 and nLedsOn <= n_target:
        fitness = (CL_star + 1) * (RLSCL + 1)
    else:
        fitness = nLedsOn * ((CL_star + 1) * (RLSCL + 1) + 1)
    
    return fitness


def local_search_hill_climbing(led_indices, powers, led_spectra, target_spectrum,
                               wavelengths, weights, n_target, led_database, 
                               led_efficiencies, pmax, max_attempts=3):
    """Post-evaluation local search (hill climbing)"""
    best_indices = led_indices.copy()
    best_powers = powers.copy()
    best_fitness = calculate_fitness_enhanced(led_indices, powers, led_spectra, 
                                            target_spectrum, wavelengths, weights, n_target)
    
    for attempt in range(max_attempts):
        improved = False
        
        # Strategy 1: Swap lowest power LED
        if len(best_indices) > 1:
            min_power_idx = np.argmin(best_powers)
            led_to_remove = best_indices[min_power_idx]
            
            # Find replacement candidates
            available = [i for i in range(len(led_database)) if i not in best_indices]
            if available:
                # Try a few random replacements
                candidates = np.random.choice(available, size=min(5, len(available)), replace=False)
                
                for candidate in candidates:
                    test_indices = best_indices.copy()
                    test_indices[min_power_idx] = candidate
                    
                    test_powers = fit_powers_nnls_enhanced(
                        test_indices, led_spectra, target_spectrum, wavelengths, 
                        weights, led_efficiencies, pmax, led_database
                    )
                    
                    if test_powers:
                        test_fitness = calculate_fitness_enhanced(
                            test_indices, test_powers, led_spectra, target_spectrum,
                            wavelengths, weights, n_target
                        )
                        
                        if test_fitness < best_fitness:
                            best_indices = test_indices
                            best_powers = test_powers
                            best_fitness = test_fitness
                            improved = True
                            break
        
        # Strategy 2: Remove lowest power LED if we have too many
        if not improved and len(best_indices) > n_target:
            min_power_idx = np.argmin(best_powers)
            test_indices = [idx for i, idx in enumerate(best_indices) if i != min_power_idx]
            test_powers = [p for i, p in enumerate(best_powers) if i != min_power_idx]
            
            if test_powers:
                test_fitness = calculate_fitness_enhanced(
                    test_indices, test_powers, led_spectra, target_spectrum,
                    wavelengths, weights, n_target
                )
                
                if test_fitness < best_fitness:
                    best_indices = test_indices
                    best_powers = test_powers
                    best_fitness = test_fitness
                    improved = True
        
        if not improved:
            break
    
    return best_indices, best_powers, best_fitness


def evaluate_individual_enhanced(combination_data: Tuple, optimizer_data: Dict):
    """Enhanced individual evaluation with local search"""
    led_indices, n_target = combination_data
    
    # Fit powers with iterative refinement
    powers = fit_powers_nnls_enhanced(
        led_indices,
        optimizer_data['led_spectra'],
        optimizer_data['target_spectrum'],
        optimizer_data['wavelengths'],
        optimizer_data['weights'],
        optimizer_data['led_efficiencies'],
        optimizer_data['pmax'],
        optimizer_data['led_database']
    )
    
    if not powers:
        return led_indices, [], float('inf')
    
    # Calculate fitness
    fitness = calculate_fitness_enhanced(
        led_indices, powers,
        optimizer_data['led_spectra'],
        optimizer_data['target_spectrum'],
        optimizer_data['wavelengths'],
        optimizer_data['weights'],
        n_target
    )
    
    # Apply local search
    led_indices, powers, fitness = local_search_hill_climbing(
        led_indices, powers,
        optimizer_data['led_spectra'],
        optimizer_data['target_spectrum'],
        optimizer_data['wavelengths'],
        optimizer_data['weights'],
        n_target,
        optimizer_data['led_database'],
        optimizer_data['led_efficiencies'],
        optimizer_data['pmax']
    )
    
    return led_indices, powers, fitness


# ============================================================================
# ENHANCED MAIN OPTIMIZER CLASS
# ============================================================================

class EnhancedSolarSimulatorEA:
    """Enhanced LED Solar Simulator with Advanced EA Improvements"""
    
    def __init__(self, 
                 led_database: List[LEDSpec],
                 target_spectrum_file: str = None,
                 wavelength_range: Tuple[float, float] = (300, 1200),
                 spectral_resolution: float = 1.0,
                 population_size: int = 50,
                 max_leds: int = 40,
                 min_leds: int = 10,
                 max_total_power: float = 150.0,
                 led_to_target_distance: float = 0.10,
                 target_area: float = 0.0441,
                 n_jobs: int = -1):
        
        self.led_database = led_database
        self.wavelength_range = wavelength_range
        self.spectral_resolution = spectral_resolution
        self.population_size = population_size
        self.max_leds = max_leds
        self.min_leds = min_leds
        self.max_total_power = max_total_power
        
        # Geometry
        self.led_to_target_distance = led_to_target_distance
        self.target_area = target_area
        self.flux_to_irradiance = 1.0 / (np.pi * led_to_target_distance**2)
        
        # Parallel processing
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        print(f"🚀 Enhanced parallel processing: {self.n_jobs} cores")
        
        # Wavelength grid
        self.wavelengths = np.arange(
            wavelength_range[0], 
            wavelength_range[1] + spectral_resolution, 
            spectral_resolution
        )
        
        # Load and process data
        print("📊 Loading target spectrum...")
        self.target_spectrum = self.load_target_spectrum(target_spectrum_file)
        
        print("⚖️  Building spectral weights...")
        self.weights = self.build_weights()
        
        print("🔄 Interpolating LED spectra...")
        self.interpolate_led_spectra()
        
        print("📐 Converting to irradiance units...")
        self.convert_leds_to_irradiance()
        
        print("📈 Calculating LED characteristics...")
        self.calculate_led_characteristics()
        
        print("🔗 Building LED clusters...")
        self.build_led_clusters()
        
        self.pmax = 10.0
        self.fitness_history = []
        self.multi_target_results = {}
        
        # Enhanced tracking
        self.diversity_history = []
        self.dynamic_weights = self.weights.copy()
        
        print("✅ Enhanced optimizer initialized!")
    
    def load_target_spectrum(self, filename: str = None) -> np.ndarray:
        """Load target spectrum (AM1.5G)"""
        if filename and os.path.exists(filename):
            try:
                file_ext = Path(filename).suffix.lower()
                
                if file_ext in ('.xlsx', '.xls'):
                    data = pd.read_excel(filename)
                else:
                    data = pd.read_csv(filename)
                
                wavelengths = data.iloc[:, 0].values
                irradiance = data.iloc[:, 1].values
                
                valid_mask = ~(np.isnan(wavelengths) | np.isnan(irradiance))
                wavelengths = wavelengths[valid_mask]
                irradiance = irradiance[valid_mask]
                
                interp_func = interpolate.interp1d(
                    wavelengths, irradiance, 
                    bounds_error=False, fill_value=0.0
                )
                interpolated = interp_func(self.wavelengths)
                
                # Normalize to 1000 W/m²
                area = np.trapz(interpolated, self.wavelengths)
                if area > 0:
                    interpolated *= (1000.0 / area)
                
                print(f"  ✅ Loaded: {filename}")
                return interpolated
            except Exception as e:
                print(f"  ⚠️  Error loading {filename}: {e}")
        
        print("  ⚠️  Using built-in AM1.5G approximation")
        return self.standard_am15g_spectrum()
    
    def standard_am15g_spectrum(self) -> np.ndarray:
        """Built-in AM1.5G approximation"""
        wavelengths = self.wavelengths
        spectrum = np.zeros_like(wavelengths)
        
        # UV
        uv_mask = (wavelengths >= 300) & (wavelengths <= 400)
        spectrum[uv_mask] = (
            0.15 * np.exp(-(wavelengths[uv_mask] - 340)**2 / (2 * 25**2)) +
            0.25 * np.exp(-(wavelengths[uv_mask] - 370)**2 / (2 * 20**2))
        )
        
        # Visible
        vis_mask = (wavelengths >= 400) & (wavelengths <= 700)
        spectrum[vis_mask] = (
            0.9 * np.exp(-(wavelengths[vis_mask] - 450)**2 / (2 * 40**2)) +
            1.4 * np.exp(-(wavelengths[vis_mask] - 500)**2 / (2 * 50**2)) +
            1.5 * np.exp(-(wavelengths[vis_mask] - 550)**2 / (2 * 60**2)) +
            1.2 * np.exp(-(wavelengths[vis_mask] - 600)**2 / (2 * 50**2)) +
            0.9 * np.exp(-(wavelengths[vis_mask] - 650)**2 / (2 * 40**2))
        )
        
        # NIR
        nir1_mask = (wavelengths >= 700) & (wavelengths <= 1000)
        spectrum[nir1_mask] = 0.7 * np.exp(-(wavelengths[nir1_mask] - 800)**2 / (2 * 80**2))
        
        nir2_mask = (wavelengths >= 1000) & (wavelengths <= 1200)
        spectrum[nir2_mask] = 0.4 * np.exp(-(wavelengths[nir2_mask] - 1100)**2 / (2 * 120**2))
        
        # Normalize
        total = np.trapz(spectrum, wavelengths)
        if total > 0:
            spectrum *= (1000.0 / total)
        
        return spectrum
    
    def build_weights(self) -> np.ndarray:
        """Build spectral weights with emphasis on 600-800nm"""
        w = self.target_spectrum.copy()
        w /= (w.max() + 1e-12)
        
        # Region-specific weights
        critical_regions = [
            (300, 400, 1.0),
            (400, 500, 2.0),
            (500, 600, 2.5),
            (600, 700, 5.0),  # Critical
            (700, 800, 5.0),  # Critical
            (800, 900, 3.0),
            (900, 1000, 2.5),
            (1000, 1100, 2.0),
            (1100, 1200, 2.0)
        ]
        
        for λ_start, λ_end, boost in critical_regions:
            mask = (self.wavelengths >= λ_start) & (self.wavelengths <= λ_end)
            w[mask] *= boost
        
        return w
    
    def interpolate_led_spectra(self):
        """Interpolate all LED spectra"""
        self.led_spectra = []
        
        for led in self.led_database:
            interp_func = interpolate.interp1d(
                led.wavelengths, led.intensities,
                kind='linear', bounds_error=False, fill_value=0.0
            )
            interpolated = interp_func(self.wavelengths)
            interpolated = np.maximum(interpolated, 0)
            self.led_spectra.append(interpolated)
        
        self.led_spectra = np.array(self.led_spectra)
    
    def convert_leds_to_irradiance(self):
        """Convert LED SPDs to irradiance"""
        for i in range(len(self.led_spectra)):
            self.led_spectra[i] = self.led_spectra[i] * self.flux_to_irradiance
    
    def calculate_led_characteristics(self):
        """Calculate LED efficiencies"""
        self.led_efficiencies = []
        
        for led_spectrum in self.led_spectra:
            total_power = np.trapz(led_spectrum, self.wavelengths)
            self.led_efficiencies.append(total_power)
    
    def build_led_clusters(self, wavelength_tolerance=10.0):
        """Build LED clusters for similar LEDs"""
        self.led_clusters = defaultdict(list)
        
        for i, led in enumerate(self.led_database):
            # Find cluster based on peak wavelength
            cluster_key = round(led.peak_wavelength / wavelength_tolerance) * wavelength_tolerance
            self.led_clusters[cluster_key].append(i)
        
        # Remove single-LED clusters
        self.led_clusters = {k: v for k, v in self.led_clusters.items() if len(v) > 1}
        
        print(f"  📊 Created {len(self.led_clusters)} LED clusters")
    
    def update_dynamic_weights(self, cl_star_history, window_size=5):
        """Update dynamic weights based on recent CL* performance"""
        if len(cl_star_history) < window_size:
            return self.weights.copy()
        
        recent_cls = cl_star_history[-window_size:]
        avg_cl_star = np.mean(recent_cls)
        
        # If CL* is high, increase weights in problematic regions
        if avg_cl_star > 0.15:
            dynamic_weights = self.weights.copy()
            
            # Find worst spectral region
            bins = [
                (300, 400), (400, 500), (500, 600), (600, 700), (700, 800),
                (800, 900), (900, 1000), (1000, 1100), (1100, 1200)
            ]
            
            # This is a simplified approach - in practice, you'd track per-bin CL values
            # For now, boost critical regions more when overall performance is poor
            critical_regions = [(600, 700), (700, 800)]
            for λ_start, λ_end in critical_regions:
                mask = (self.wavelengths >= λ_start) & (self.wavelengths <= λ_end)
                dynamic_weights[mask] *= 1.5
            
            return dynamic_weights
        
        return self.weights.copy()
    
    def calculate_population_diversity(self, population):
        """Calculate population diversity based on LED set similarity"""
        if len(population) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                set1 = set(population[i].led_indices)
                set2 = set(population[j].led_indices)
                
                # Jaccard similarity
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        # Diversity is 1 - average similarity
        return 1.0 - np.mean(similarities)
    
    def prepare_optimizer_data(self) -> Dict:
        """Prepare data for parallel workers"""
        return {
            'led_spectra': self.led_spectra,
            'target_spectrum': self.target_spectrum,
            'wavelengths': self.wavelengths,
            'weights': self.weights,
            'led_efficiencies': self.led_efficiencies,
            'pmax': self.pmax,
            'led_database': self.led_database
        }
    
    def calculate_combined_spectrum(self, combination: LEDCombination) -> np.ndarray:
        """Calculate combined spectrum"""
        combined = np.zeros_like(self.wavelengths)
        for led_idx, power in zip(combination.led_indices, combination.powers):
            if led_idx < len(self.led_database):
                combined += power * self.led_spectra[led_idx]
        return combined
    
    def calculate_spectral_ratio_metrics(self, simulated: np.ndarray) -> Dict:
        """Calculate IEC 60904-9 metrics"""
        bins = [
            (300, 400), (400, 500), (500, 600), (600, 700), (700, 800),
            (800, 900), (900, 1000), (1000, 1100), (1100, 1200)
        ]
        
        spectral_ratios = []
        CLs = []
        
        for lambda1, lambda2 in bins:
            mask = (self.wavelengths >= lambda1) & (self.wavelengths <= lambda2)
            numerator = np.trapz(simulated[mask], self.wavelengths[mask])
            denominator = np.trapz(self.target_spectrum[mask], self.wavelengths[mask])
            
            SR = numerator / denominator if denominator > 1e-12 else 0.0
            CL = abs(1.0 - SR)
            
            spectral_ratios.append(SR)
            CLs.append(CL)
        
        CL_star = max(CLs)
        classification = 'A+' if CL_star <= 0.05 else 'A' if CL_star <= 0.25 else 'B' if CL_star <= 0.4 else 'C'
        
        return {
            'spectral_ratios': spectral_ratios,
            'CLs': CLs,
            'CL_star': CL_star,
            'classification': classification,
            'bins': bins
        }
    
    def calculate_relative_least_squares(self, simulated: np.ndarray) -> float:
        """Calculate RLS"""
        denominator = self.target_spectrum + 1e-12
        numerator = (self.target_spectrum - simulated) ** 2
        rls_values = numerator / denominator
        weighted_rls = rls_values * self.weights
        RLS = np.sum(weighted_rls) / np.sum(self.weights)
        return RLS
    
    # ========================================================================
    # ENHANCED EA WITH IMPROVEMENTS
    # ========================================================================
    
    def initialize_population_enhanced(self, n_target: int, pop_size: int = None) -> List:
        """Enhanced initialization with multiple strategies"""
        if pop_size is None:
            pop_size = self.population_size
        
        print(f"🔄 Enhanced initialization ({pop_size} individuals)...")
        
        candidates = []
        
        # Strategy 1: Multiple greedy seeds (5 different strategies)
        print("  📍 Creating 5 greedy seeds...")
        for i in range(5):
            seed = self.generate_greedy_seed_variant(n_target, variant=i)
            if seed:
                candidates.append((seed, n_target))
        
        # Strategy 2: Coverage-based (spread across spectrum)
        print("  📍 Creating 5 coverage-based seeds...")
        for _ in range(5):
            seed = self.generate_coverage_seed(n_target)
            candidates.append((seed, n_target))
        
        # Strategy 3: Cluster-based (one LED per cluster)
        print("  📍 Creating 5 cluster-based seeds...")
        for _ in range(5):
            seed = self.generate_cluster_seed(n_target)
            candidates.append((seed, n_target))
        
        # Strategy 4: Region-balanced (35% of population)
        n_balanced = int(pop_size * 0.35)
        print(f"  📍 Creating {n_balanced} region-balanced seeds...")
        for _ in range(n_balanced):
            seed = self.generate_region_balanced_seed(n_target)
            candidates.append((seed, n_target))
        
        # Strategy 5: Fill rest with random
        n_random = pop_size - len(candidates)
        print(f"  📍 Creating {n_random} random seeds...")
        for _ in range(n_random):
            seed = self.generate_random_seed(n_target)
            candidates.append((seed, n_target))
        
        # Parallel evaluation with enhanced functions
        optimizer_data = self.prepare_optimizer_data()
        
        results = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(evaluate_individual_enhanced)(cand, optimizer_data)
            for cand in tqdm(candidates, desc="Evaluating", disable=len(candidates)<20)
        )
        
        population = []
        for led_indices, powers, fitness in results:
            combo = LEDCombination(led_indices, powers, fitness, n_target)
            population.append(combo)
        
        population.sort(key=lambda x: x.fitness)
        print(f"  ✅ Best initial: {population[0].fitness:.6f}")
        
        return population

    def generate_greedy_seed_variant(self, n_target: int, variant: int = 0) -> List[int]:
        """Multiple greedy strategies"""
        regions = [
            (300, 400, 0.08), (400, 500, 0.12), (500, 600, 0.15),
            (600, 700, 0.20), (700, 800, 0.20), (800, 900, 0.12),
            (900, 1000, 0.08), (1000, 1100, 0.03), (1100, 1200, 0.02)
        ]
        
        # Boost critical regions for some variants
        if variant == 1:  # Emphasize 600-800nm
            regions[3] = (600, 700, 0.25)
            regions[4] = (700, 800, 0.25)
        elif variant == 2:  # Emphasize visible
            regions[2] = (500, 600, 0.20)
            regions[3] = (600, 700, 0.25)
        elif variant == 3:  # Emphasize NIR
            regions[4] = (700, 800, 0.25)
            regions[5] = (800, 900, 0.15)
        elif variant == 4:  # Balanced
            pass  # Use default
        
        selected = []
        for wl_min, wl_max, fraction in regions:
            n_from_region = max(1, int(n_target * fraction))
            candidates = [
                (i, self.led_efficiencies[i])
                for i, led in enumerate(self.led_database)
                if wl_min <= led.peak_wavelength < wl_max and i not in selected
            ]
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                # Take top N with some randomness
                n_take = min(n_from_region, len(candidates))
                if variant == 0:
                    chosen = [idx for idx, _ in candidates[:n_take]]
                else:
                    # Add some randomness for diversity
                    n_best = max(1, n_take // 2)
                    chosen = [idx for idx, _ in candidates[:n_best]]
                    if len(candidates) > n_best and n_take > n_best:
                        extra = np.random.choice(
                            [idx for idx, _ in candidates[n_best:]], 
                            size=min(n_take - n_best, len(candidates) - n_best),
                            replace=False
                        ).tolist()
                        chosen.extend(extra)
                selected.extend(chosen)
        
        return selected[:n_target]

    def generate_coverage_seed(self, n_target: int) -> List[int]:
        """Maximize wavelength coverage"""
        wl_min, wl_max = self.wavelength_range
        bin_width = (wl_max - wl_min) / n_target
        
        selected = []
        for i in range(n_target):
            bin_start = wl_min + i * bin_width
            bin_end = bin_start + bin_width
            
            candidates = [
                idx for idx, led in enumerate(self.led_database)
                if bin_start <= led.peak_wavelength < bin_end and idx not in selected
            ]
            
            if candidates:
                # Choose best efficiency in bin
                best = max(candidates, key=lambda i: self.led_efficiencies[i])
                selected.append(best)
        
        return selected[:n_target]

    def generate_cluster_seed(self, n_target: int) -> List[int]:
        """Generate seed using LED clusters"""
        selected = []
        cluster_keys = list(self.led_clusters.keys())
        
        # Select one LED from each cluster
        for cluster_key in cluster_keys:
            if len(selected) >= n_target:
                break
            cluster_leds = self.led_clusters[cluster_key]
            # Choose best efficiency in cluster
            best = max(cluster_leds, key=lambda i: self.led_efficiencies[i])
            selected.append(best)
        
        # Fill remaining slots with random LEDs
        while len(selected) < n_target:
            available = [i for i in range(len(self.led_database)) if i not in selected]
            if available:
                selected.append(np.random.choice(available))
            else:
                break
        
        return selected[:n_target]

    def generate_region_balanced_seed(self, n_target: int) -> List[int]:
        """Generate region-balanced seed"""
        regions = [
            (300, 400, 0.08), (400, 500, 0.12), (500, 600, 0.15),
            (600, 700, 0.20), (700, 800, 0.20), (800, 900, 0.12),
            (900, 1000, 0.08), (1000, 1100, 0.03), (1100, 1200, 0.02)
        ]
        
        selected = []
        for wl_min, wl_max, fraction in regions:
            n_from_region = max(1, int(n_target * fraction))
            candidates = [
                i for i, led in enumerate(self.led_database)
                if wl_min <= led.peak_wavelength < wl_max and i not in selected
            ]
            if candidates:
                chosen = np.random.choice(
                    candidates,
                    size=min(n_from_region, len(candidates)),
                    replace=False
                ).tolist()
                selected.extend(chosen)
        
        while len(selected) < n_target:
            available = [i for i in range(len(self.led_database)) if i not in selected]
            if available:
                selected.append(np.random.choice(available))
            else:
                break
        
        return selected[:n_target]
    
    def generate_random_seed(self, n_target: int) -> List[int]:
        """Generate random seed"""
        n_leds = np.random.randint(
            max(self.min_leds, n_target - 3),
            min(n_target + 2, len(self.led_database))
        )
        return np.random.choice(
            len(self.led_database),
            size=n_leds,
            replace=False
        ).tolist()

    def targeted_mutation(self, indices, current_best, n_target):
        """Enhanced targeted mutation targeting weak spectral regions"""
        # Calculate which regions are weak
        combined = self.calculate_combined_spectrum(current_best)
        sr_metrics = self.calculate_spectral_ratio_metrics(combined)
        
        # Find worst region
        worst_region_idx = np.argmax(sr_metrics['CLs'])
        worst_bin = sr_metrics['bins'][worst_region_idx]
        
        mutation_type = np.random.choice(['add_to_weak', 'remove_weak', 'swap_in_weak', 'remove_lowest_power'])
        
        if mutation_type == 'add_to_weak' and len(indices) < n_target:
            # Add LED to weak region
            candidates = [
                i for i in range(len(self.led_database))
                if i not in indices 
                and worst_bin[0] <= self.led_database[i].peak_wavelength < worst_bin[1]
            ]
            if candidates:
                best = max(candidates, key=lambda i: self.led_efficiencies[i])
                indices.append(best)
            else:
                # Fallback: add any available LED
                available = [i for i in range(len(self.led_database)) if i not in indices]
                if available:
                    indices.append(np.random.choice(available))
        
        elif mutation_type == 'remove_weak' and len(indices) > self.min_leds:
            # Remove LED from weak region
            weak_region_leds = [
                i for i, idx in enumerate(indices)
                if worst_bin[0] <= self.led_database[idx].peak_wavelength < worst_bin[1]
            ]
            if weak_region_leds:
                indices.pop(np.random.choice(weak_region_leds))
            else:
                # Fallback: remove random
                if len(indices) > 1:
                    indices.pop(np.random.randint(len(indices)))
        
        elif mutation_type == 'swap_in_weak':
            # Swap in a better LED for weak region
            if len(indices) > 0:
                idx_to_replace = np.random.randint(len(indices))
                candidates = [
                    i for i in range(len(self.led_database))
                    if i not in indices
                    and worst_bin[0] <= self.led_database[i].peak_wavelength < worst_bin[1]
                ]
                if candidates:
                    best = max(candidates, key=lambda i: self.led_efficiencies[i])
                    indices[idx_to_replace] = best
        
        elif mutation_type == 'remove_lowest_power' and len(indices) > self.min_leds:
            # Remove LED with lowest power (if we have power info)
            if hasattr(current_best, 'powers') and current_best.powers:
                # Find LED with lowest power in current best
                min_power_idx = np.argmin(current_best.powers)
                if min_power_idx < len(indices):
                    # Find corresponding LED in current individual
                    led_to_remove = current_best.led_indices[min_power_idx]
                    if led_to_remove in indices:
                        indices.remove(led_to_remove)
            else:
                # Fallback: remove random
                if len(indices) > 1:
                    indices.pop(np.random.randint(len(indices)))
        
        return indices

    def weighted_crossover(self, p1, p2, n_target):
        """Weighted crossover based on parent fitness and LED power factors"""
        # Combine parent LED sets
        combined = list(set(p1.led_indices + p2.led_indices))
        
        if len(combined) <= n_target:
            return combined
        
        # Calculate weights for each LED
        led_weights = {}
        
        # Weight based on parent fitness (lower fitness = higher weight)
        p1_weight = 1.0 / (p1.fitness + 1e-6)
        p2_weight = 1.0 / (p2.fitness + 1e-6)
        
        for led_idx in combined:
            weight = 0.0
            
            # Weight from parent 1
            if led_idx in p1.led_indices:
                # Find power factor in parent 1
                p1_power_idx = p1.led_indices.index(led_idx)
                if p1_power_idx < len(p1.powers):
                    power_factor = p1.powers[p1_power_idx]
                    weight += p1_weight * (power_factor + 0.1)  # Add small constant to avoid zero weights
            
            # Weight from parent 2
            if led_idx in p2.led_indices:
                p2_power_idx = p2.led_indices.index(led_idx)
                if p2_power_idx < len(p2.powers):
                    power_factor = p2.powers[p2_power_idx]
                    weight += p2_weight * (power_factor + 0.1)
            
            # Add efficiency weight
            if led_idx < len(self.led_efficiencies):
                weight += self.led_efficiencies[led_idx] * 0.1
            
            led_weights[led_idx] = weight
        
        # Select LEDs based on weights
        led_list = list(led_weights.keys())
        weights_list = [led_weights[led] for led in led_list]
        
        # Normalize weights
        weights_list = np.array(weights_list)
        weights_list = weights_list / (np.sum(weights_list) + 1e-12)
        
        # Select n_target LEDs
        selected_indices = np.random.choice(
            len(led_list), 
            size=min(n_target, len(led_list)), 
            replace=False, 
            p=weights_list
        )
        
        return [led_list[i] for i in selected_indices]

    def evolutionary_algorithm_enhanced(self, 
                                    n_target: int,
                                    pop_size: int = 50,
                                    n_generations: int = 100,
                                    elite_size: int = 5,
                                    tournament_size: int = 3,
                                    crossover_rate: float = 0.7,
                                    mutation_rate: float = 0.3) -> LEDCombination:
        """Enhanced EA with all improvements"""
        print(f"\n{'='*70}")
        print(f"🧬 ENHANCED EA: {n_target} LEDs | {self.n_jobs} cores")
        print(f"{'='*70}")
        
        # Initialize with enhanced strategies
        population = self.initialize_population_enhanced(n_target, pop_size)
        best_ever = min(population, key=lambda x: x.fitness)
        
        fitness_history = []
        cl_star_history = []
        rls_history = []
        diversity_history = []
        
        combined = self.calculate_combined_spectrum(best_ever)
        sr_metrics = self.calculate_spectral_ratio_metrics(combined)
        rls = self.calculate_relative_least_squares(combined)
        
        fitness_history.append(best_ever.fitness)
        cl_star_history.append(sr_metrics['CL_star'])
        rls_history.append(rls)
        diversity_history.append(self.calculate_population_diversity(population))
        
        print(f"\n📊 Initial: Fitness={best_ever.fitness:.6f} | CL*={sr_metrics['CL_star']:.4f} | RLS={rls:.4f}")
        print(f"\n{'Gen':<5} {'Fitness':<12} {'CL*':<10} {'RLS':<10} {'LEDs':<6} {'Div':<6} {'Status'}")
        print("-"*80)
        
        optimizer_data = self.prepare_optimizer_data()
        
        # Adaptive parameters
        no_improvement_count = 0
        current_mutation_rate = mutation_rate
        stagnation_threshold = 25  # Generations without improvement
        
        for generation in range(n_generations):
            population.sort(key=lambda x: x.fitness)
            current_best = population[0]
            
            combined = self.calculate_combined_spectrum(current_best)
            sr_metrics = self.calculate_spectral_ratio_metrics(combined)
            rls = self.calculate_relative_least_squares(combined)
            
            fitness_history.append(current_best.fitness)
            cl_star_history.append(sr_metrics['CL_star'])
            rls_history.append(rls)
            diversity_history.append(self.calculate_population_diversity(population))
            
            # Update dynamic weights
            self.dynamic_weights = self.update_dynamic_weights(cl_star_history)
            
            # Check improvement
            improved = False
            if current_best.fitness < best_ever.fitness * 0.999:  # 0.1% improvement threshold
                best_ever = current_best
                improved = True
                no_improvement_count = 0
                current_mutation_rate = mutation_rate  # Reset mutation
            else:
                no_improvement_count += 1
            
            # ADAPTIVE MUTATION: Increase when stuck
            if no_improvement_count > 10:
                current_mutation_rate = min(0.6, mutation_rate * 1.5)
            elif no_improvement_count > 20:
                current_mutation_rate = min(0.8, mutation_rate * 2.0)
            
            # Print progress
            print_freq = 1 if n_generations <= 50 else 5
            
            if generation % print_freq == 0 or improved or generation == n_generations - 1:
                status = "★ NEW BEST" if improved else f"(stall: {no_improvement_count})" if no_improvement_count > 10 else ""
                
                if sr_metrics['CL_star'] <= 0.05:
                    cl_indicator = "✓✓"
                elif sr_metrics['CL_star'] <= 0.10:
                    cl_indicator = "✓ "
                elif sr_metrics['CL_star'] <= 0.25:
                    cl_indicator = "○ "
                else:
                    cl_indicator = "✗ "
                
                diversity = diversity_history[-1]
                print(f"{generation:<5} {current_best.fitness:<12.6f} "
                    f"{sr_metrics['CL_star']:<10.4f} {rls:<10.4f} "
                    f"{len(current_best.led_indices):<6} {diversity:<6.3f} {cl_indicator} {status}")
            
            # ENHANCED STAGNATION DETECTION
            if no_improvement_count > stagnation_threshold:
                print(f"   🔄 RESTART: No improvement for {stagnation_threshold} generations. Injecting diversity...")
                
                # Check diversity
                current_diversity = diversity_history[-1]
                if current_diversity < 0.3:  # Low diversity
                    print(f"   📊 Low diversity detected ({current_diversity:.3f}). Full restart...")
                    # Keep only top 10%, replace rest
                    n_keep = max(1, pop_size // 10)
                else:
                    # Keep top 20%, replace rest
                    n_keep = pop_size // 5
                
                population = population[:n_keep]
                
                new_individuals = []
                for _ in range(pop_size - n_keep):
                    seed = self.generate_random_seed(n_target)
                    new_individuals.append((seed, n_target))
                
                results = Parallel(n_jobs=self.n_jobs, backend='loky')(
                    delayed(evaluate_individual_enhanced)(ind, optimizer_data)
                    for ind in new_individuals
                )
                
                for led_indices, powers, fitness in results:
                    combo = LEDCombination(led_indices, powers, fitness, n_target)
                    population.append(combo)
                
                no_improvement_count = 0
                current_mutation_rate = mutation_rate * 1.2
            
            # Elitism
            next_population = population[:elite_size]
            
            # Generate offspring
            n_offspring = pop_size - elite_size
            offspring_data = []
            
            for _ in range(n_offspring):
                p1 = self.tournament_select(population, tournament_size)
                p2 = self.tournament_select(population, tournament_size)
                
                if np.random.random() < crossover_rate:
                    child = self.weighted_crossover(p1, p2, n_target)
                else:
                    child = p1.led_indices.copy()
                
                if np.random.random() < current_mutation_rate:
                    # Enhanced targeted mutation
                    child = self.targeted_mutation(child, current_best, n_target)
                
                offspring_data.append((child, n_target))
            
            # Parallel evaluation with enhanced functions
            offspring_results = Parallel(n_jobs=self.n_jobs, backend='loky')(
                delayed(evaluate_individual_enhanced)(data, optimizer_data)
                for data in offspring_data
            )
            
            for led_indices, powers, fitness in offspring_results:
                combo = LEDCombination(led_indices, powers, fitness, n_target)
                next_population.append(combo)
            
            population = next_population
        
        # Final summary
        print("="*80)
        combined = self.calculate_combined_spectrum(best_ever)
        sr_metrics = self.calculate_spectral_ratio_metrics(combined)
        rls = self.calculate_relative_least_squares(combined)
        
        print(f"\n🏆 Best: Fitness={best_ever.fitness:.6f} | CL*={sr_metrics['CL_star']:.4f} | Class={sr_metrics['classification']}")
        
        # Show problem regions
        print(f"\n📊 Regional CL values:")
        regions = ['UV', 'Blue', 'Green', 'Red', 'NIR1', 'NIR2', 'NIR3', 'NIR4', 'NIR5']
        for i, (region, cl) in enumerate(zip(regions, sr_metrics['CLs'])):
            indicator = "✓" if cl <= 0.10 else "⚠️" if cl <= 0.25 else "✗"
            print(f"   {region:<8}: {cl:.4f} {indicator}")
        
        self.fitness_history = fitness_history
        self.cl_star_history = cl_star_history
        self.rls_history = rls_history
        self.diversity_history = diversity_history
        
        return best_ever
    
    def tournament_select(self, population, size):
        """Tournament selection"""
        indices = np.random.choice(len(population), size=size, replace=False)
        tournament = [population[i] for i in indices]
        return min(tournament, key=lambda x: x.fitness)
    
    def optimize_multi_target_enhanced(self, 
                                    target_led_counts: List[int],
                                    sequential: bool = False) -> Dict:
        """Enhanced multi-target optimization"""
        print("\n" + "="*70)
        print("🎯 ENHANCED MULTI-TARGET OPTIMIZATION")
        print("="*70)
        print(f"Targets: {target_led_counts}")
        print(f"Mode: {'Sequential' if sequential else 'Parallel (ALL targets simultaneously)'}")
        
        if sequential:
            results = self._optimize_sequential_enhanced(target_led_counts)
        else:
            results = self._optimize_truly_parallel_enhanced(target_led_counts)
        
        # Find best
        sorted_results = sorted(results.items(), 
                            key=lambda x: (x[1]['CL_star'], x[1]['fitness']))
        best_target = sorted_results[0][0]
        
        self.multi_target_results = {
            'all_results': results,
            'best_target': best_target,
            'best_solution': results[best_target]['solution']
        }
        
        return self.multi_target_results

    def _optimize_sequential_enhanced(self, target_led_counts):
        """Sequential optimization with enhanced EA"""
        results = {}
        for n_target in target_led_counts:
            params = self.get_adaptive_ea_parameters(n_target)
            solution = self.evolutionary_algorithm_enhanced(
                n_target=n_target,
                pop_size=params['pop_size'],
                n_generations=params['n_generations'],
                elite_size=params['elite_size'],
                tournament_size=params['tournament_size'],
                crossover_rate=params['crossover_rate'],
                mutation_rate=params['mutation_rate']
            )
            results[n_target] = self._package_result(solution, n_target)
        return results

    def _optimize_truly_parallel_enhanced(self, target_led_counts):
        """Parallel optimization with enhanced EA"""
        print(f"\n⚡ Running {len(target_led_counts)} targets SIMULTANEOUSLY")
        print(f"   Using ~{self.n_jobs // len(target_led_counts)} cores per target")
        print(f"   This should be ~{len(target_led_counts)}x faster!\n")
        
        def optimize_one_target_enhanced(n_target, cores_per_target):
            """Run enhanced optimization for one target"""
            import copy
            ea_instance = copy.copy(self)
            ea_instance.n_jobs = cores_per_target
            
            params = self.get_adaptive_ea_parameters(n_target)
            solution = ea_instance.evolutionary_algorithm_enhanced(
                n_target=n_target,
                pop_size=params['pop_size'],
                n_generations=params['n_generations'],
                elite_size=params['elite_size'],
                tournament_size=params['tournament_size'],
                crossover_rate=params['crossover_rate'],
                mutation_rate=params['mutation_rate']
            )
            return n_target, solution
        
        # Distribute cores among targets
        cores_per_target = max(2, self.n_jobs // len(target_led_counts))
        
        # Run ALL targets in parallel
        from joblib import Parallel, delayed
        target_results = Parallel(n_jobs=len(target_led_counts), backend='loky')(
            delayed(optimize_one_target_enhanced)(n_target, cores_per_target)
            for n_target in target_led_counts
        )
        
        # Package results
        results = {}
        for n_target, solution in target_results:
            results[n_target] = self._package_result(solution, n_target)
        
        return results

    def _package_result(self, solution, n_target):
        """Package solution with metrics"""
        combined = self.calculate_combined_spectrum(solution)
        sr_metrics = self.calculate_spectral_ratio_metrics(combined)
        RLS = self.calculate_relative_least_squares(combined)
        
        return {
            'solution': solution,
            'fitness': solution.fitness,
            'actual_led_count': len(solution.led_indices),
            'RLS': RLS,
            'CL_star': sr_metrics['CL_star'],
            'classification': sr_metrics['classification'],
            'sr_metrics': sr_metrics
        }
    
    def get_adaptive_ea_parameters(self, n_target: int) -> Dict:
        """Get EA parameters based on target"""
        if n_target <= 25:
            return {'pop_size': 70, 'n_generations': 120, 'elite_size': 6,
                    'tournament_size': 3, 'crossover_rate': 0.7, 'mutation_rate': 0.30}
        elif n_target <= 27:
            return {'pop_size': 80, 'n_generations': 140, 'elite_size': 7,
                    'tournament_size': 3, 'crossover_rate': 0.7, 'mutation_rate': 0.27}
        else:
            return {'pop_size': 90, 'n_generations': 160, 'elite_size': 8,
                    'tournament_size': 3, 'crossover_rate': 0.7, 'mutation_rate': 0.22}
    
    def plot_enhanced_evolution_history(self, save_path: str = None):
        """Plot enhanced evolution history with diversity"""
        if not hasattr(self, 'cl_star_history'):
            print("No evolution history available")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        generations = range(len(self.fitness_history))
        
        # Plot 1: Fitness evolution
        ax1 = axes[0]
        ax1.plot(generations, self.fitness_history, 'b-', linewidth=2, label='Fitness')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness (lower is better)')
        ax1.set_title('Fitness Evolution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: CL* and RLS evolution
        ax2 = axes[1]
        ax2.plot(generations, self.cl_star_history, 'r-', linewidth=2, label='CL*')
        ax2.plot(generations, self.rls_history, 'g-', linewidth=2, label='RLS')
        
        # Add target lines
        ax2.axhline(y=0.05, color='green', linestyle='--', linewidth=1, alpha=0.5, label='A+ (CL*≤0.05)')
        ax2.axhline(y=0.10, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Target (CL*≤0.10)')
        ax2.axhline(y=0.25, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Class A (CL*≤0.25)')
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Value')
        ax2.set_title('Spectral Match Quality Evolution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Population diversity
        ax3 = axes[2]
        ax3.plot(generations, self.diversity_history, 'purple', linewidth=2, label='Population Diversity')
        ax3.axhline(y=0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Low Diversity Threshold')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Diversity (0-1)')
        ax3.set_title('Population Diversity Evolution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Enhanced evolution plot saved: {save_path}")
        
        plt.show()
    
    def plot_pareto_front(self, save_path: str = None):
        """Plot Pareto front for multi-objective analysis"""
        if not self.multi_target_results:
            print("No multi-target results available")
            return
        
        results = self.multi_target_results['all_results']
        
        led_counts = []
        cl_stars = []
        fitnesses = []
        classifications = []
        
        for n_target, result in results.items():
            led_counts.append(result['actual_led_count'])
            cl_stars.append(result['CL_star'])
            fitnesses.append(result['fitness'])
            classifications.append(result['classification'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: LED Count vs CL*
        colors = {'A+': 'green', 'A': 'blue', 'B': 'orange', 'C': 'red'}
        for i, (led_count, cl_star, classification) in enumerate(zip(led_counts, cl_stars, classifications)):
            ax1.scatter(led_count, cl_star, c=colors.get(classification, 'gray'), 
                       s=100, alpha=0.7, label=classification if classification not in [x.get_text() for x in ax1.get_legend().get_texts()] else "")
        
        ax1.set_xlabel('Number of LEDs')
        ax1.set_ylabel('CL* (lower is better)')
        ax1.set_title('Pareto Front: LED Count vs Spectral Quality')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add classification lines
        ax1.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='A+ (CL*≤0.05)')
        ax1.axhline(y=0.25, color='blue', linestyle='--', alpha=0.5, label='A (CL*≤0.25)')
        
        # Plot 2: LED Count vs Fitness
        for i, (led_count, fitness, classification) in enumerate(zip(led_counts, fitnesses, classifications)):
            ax2.scatter(led_count, fitness, c=colors.get(classification, 'gray'), 
                       s=100, alpha=0.7, label=classification if classification not in [x.get_text() for x in ax2.get_legend().get_texts()] else "")
        
        ax2.set_xlabel('Number of LEDs')
        ax2.set_ylabel('Fitness (lower is better)')
        ax2.set_title('Pareto Front: LED Count vs Overall Fitness')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Pareto front plot saved: {save_path}")
        
        plt.show()
    
    def save_configuration(self, solution: LEDCombination, filename: str):
        """Save configuration to JSON"""
        combined = self.calculate_combined_spectrum(solution)
        sr_metrics = self.calculate_spectral_ratio_metrics(combined)
        RLS = self.calculate_relative_least_squares(combined)
        
        config = {
            'led_configuration': [],
            'RLS': float(RLS),
            'CL_star': float(sr_metrics['CL_star']),
            'classification': sr_metrics['classification']
        }
        
        for led_idx, power in zip(solution.led_indices, solution.powers):
            led = self.led_database[led_idx]
            config['led_configuration'].append({
                'name': led.name,
                'peak_wavelength': float(led.peak_wavelength),
                'viewing_angle': float(led.viewing_angle),
                'power_factor': float(power)
            })
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

    def plot_results(self, solution: LEDCombination, save_path: str = None):
        """Plot results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        combined = self.calculate_combined_spectrum(solution)
        
        # Spectrum
        ax1.plot(self.wavelengths, self.target_spectrum, 'b-', label='Target', lw=2)
        ax1.plot(self.wavelengths, combined, 'r--', label='Simulated', lw=2)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Spectral Irradiance (W/m²/nm)')
        ax1.set_title('Spectrum Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Metrics
        sr_metrics = self.calculate_spectral_ratio_metrics(combined)
        RLS = self.calculate_relative_least_squares(combined)
        
        metrics = ['RLS'] + [f'CL{i+1}' for i in range(9)]
        values = [RLS] + sr_metrics['CLs']
        colors = ['blue'] + ['red' if cl > 0.25 else 'orange' if cl > 0.10 else 'green' 
                             for cl in sr_metrics['CLs']]
        
        ax2.bar(range(len(metrics)), values, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels(metrics, rotation=45)
        ax2.set_ylabel('Value')
        ax2.set_title(f'Metrics (Classification: {sr_metrics["classification"]})')
        ax2.axhline(0.10, color='orange', ls='--', label='10%')
        ax2.axhline(0.25, color='red', ls='--', label='25%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Alias for backward compatibility
ParallelSolarSimulatorEA = EnhancedSolarSimulatorEA