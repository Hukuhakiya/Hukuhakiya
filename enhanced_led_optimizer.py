"""
Enhanced LED Solar Simulator Optimizer - Complete Parallel Version with Advanced Improvements
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
from sklearn.cluster import KMeans
from collections import defaultdict
import copy

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


class LEDCluster:
    """Represents a cluster of similar LEDs"""
    def __init__(self, led_indices: List[int], representative_idx: int, 
                 peak_wavelength: float, cluster_id: int):
        self.led_indices = led_indices
        self.representative_idx = representative_idx
        self.peak_wavelength = peak_wavelength
        self.cluster_id = cluster_id


# ============================================================================
# MODULE-LEVEL FUNCTIONS (Required for parallel processing)
# ============================================================================

def fit_powers_nnls_iterative(led_indices, led_spectra, target_spectrum, 
                             wavelengths, weights, led_efficiencies, pmax, 
                             led_database, power_threshold=0.05):
    """Enhanced NNLS with iterative refinement to remove low-power LEDs"""
    if not led_indices:
        return []
    
    current_indices = led_indices.copy()
    iteration = 0
    max_iterations = 3
    
    while iteration < max_iterations:
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
        
        # Check for low-power LEDs to remove
        low_power_mask = powers < power_threshold
        if not np.any(low_power_mask) or len(current_indices) <= 3:
            break
        
        # Remove low-power LEDs
        current_indices = [idx for i, idx in enumerate(current_indices) if not low_power_mask[i]]
        iteration += 1
    
    # Final fit with remaining LEDs
    if len(current_indices) != len(led_indices):
        A = np.stack([led_spectra[i] for i in current_indices], axis=1)
        W = np.sqrt(weights + 1e-12)[:, None]
        Aw = W * A
        bw = W[:, 0] * b
        
        max_bounds = []
        for idx in current_indices:
            peak = led_database[idx].peak_wavelength
            efficiency = led_efficiencies[idx]
            
            if 600 <= peak <= 800:
                power_multiplier = 2.0
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
    
    # Return full-length array with zeros for removed LEDs
    full_powers = [0.0] * len(led_indices)
    for i, idx in enumerate(current_indices):
        original_pos = led_indices.index(idx)
        full_powers[original_pos] = powers[i]
    
    return full_powers


def calculate_fitness_with_dynamic_weights(led_indices, powers, led_spectra, target_spectrum,
                                         wavelengths, base_weights, n_target, previous_cl_star=None,
                                         problematic_bins=None):
    """Enhanced fitness calculation with dynamic weights based on previous CL* performance"""
    # Combined spectrum
    combined = np.zeros_like(target_spectrum)
    for led_idx, power in zip(led_indices, powers):
        combined += power * led_spectra[led_idx]
    
    # Dynamic weight adjustment
    weights = base_weights.copy()
    if previous_cl_star is not None and previous_cl_star > 0.15 and problematic_bins is not None:
        # Increase weights in problematic spectral regions
        bins = [
            (300, 400), (400, 500), (500, 600), (600, 700), (700, 800),
            (800, 900), (900, 1000), (1000, 1100), (1100, 1200)
        ]
        
        for bin_idx in problematic_bins:
            if bin_idx < len(bins):
                lambda1, lambda2 = bins[bin_idx]
                mask = (wavelengths >= lambda1) & (wavelengths <= lambda2)
                weights[mask] *= 2.0  # Double the weight in problematic regions
    
    # RLS calculation with dynamic weights
    denominator = target_spectrum + 1e-12
    numerator = (target_spectrum - combined) ** 2
    rls_values = numerator / denominator
    weighted_rls = rls_values * weights
    RLS = np.sum(weighted_rls) / np.sum(weights)
    
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
    nLedsOn = len([p for p in powers if p > 1e-6])  # Count only active LEDs
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
    
    return fitness, CLs


def local_search_hill_climbing(led_indices, powers, led_spectra, target_spectrum,
                              wavelengths, weights, n_target, led_database,
                              led_efficiencies, pmax, max_iterations=3):
    """Post-evaluation local search using hill climbing"""
    current_fitness, current_cls = calculate_fitness_with_dynamic_weights(
        led_indices, powers, led_spectra, target_spectrum, wavelengths, 
        weights, n_target
    )
    
    best_indices = led_indices.copy()
    best_powers = powers.copy()
    best_fitness = current_fitness
    
    for iteration in range(max_iterations):
        improved = False
        
        # Strategy 1: Try removing the LED with lowest power
        if len(led_indices) > 3:
            active_powers = [(i, p) for i, p in enumerate(powers) if p > 1e-6]
            if active_powers:
                min_power_idx = min(active_powers, key=lambda x: x[1])[0]
                test_indices = [idx for i, idx in enumerate(led_indices) if i != min_power_idx]
                
                if test_indices:
                    test_powers = fit_powers_nnls_iterative(
                        test_indices, led_spectra, target_spectrum, wavelengths,
                        weights, led_efficiencies, pmax, led_database
                    )
                    
                    test_fitness, _ = calculate_fitness_with_dynamic_weights(
                        test_indices, test_powers, led_spectra, target_spectrum,
                        wavelengths, weights, n_target
                    )
                    
                    if test_fitness < best_fitness:
                        best_indices = test_indices
                        best_powers = test_powers
                        best_fitness = test_fitness
                        improved = True
        
        # Strategy 2: Try swapping one LED for a better one
        if not improved and len(led_indices) > 0:
            available_leds = [i for i in range(len(led_database)) if i not in led_indices]
            if available_leds:
                # Try swapping the LED with lowest power
                active_powers = [(i, p) for i, p in enumerate(powers) if p > 1e-6]
                if active_powers:
                    swap_idx = min(active_powers, key=lambda x: x[1])[0]
                    
                    # Try a few random available LEDs
                    candidates = np.random.choice(available_leds, 
                                                size=min(3, len(available_leds)), 
                                                replace=False)
                    
                    for new_led in candidates:
                        test_indices = led_indices.copy()
                        test_indices[swap_idx] = new_led
                        
                        test_powers = fit_powers_nnls_iterative(
                            test_indices, led_spectra, target_spectrum, wavelengths,
                            weights, led_efficiencies, pmax, led_database
                        )
                        
                        test_fitness, _ = calculate_fitness_with_dynamic_weights(
                            test_indices, test_powers, led_spectra, target_spectrum,
                            wavelengths, weights, n_target
                        )
                        
                        if test_fitness < best_fitness:
                            best_indices = test_indices
                            best_powers = test_powers
                            best_fitness = test_fitness
                            improved = True
                            break
        
        if not improved:
            break
        
        # Update for next iteration
        led_indices = best_indices.copy()
        powers = best_powers.copy()
    
    return best_indices, best_powers, best_fitness


def evaluate_individual_enhanced(combination_data: Tuple, optimizer_data: Dict):
    """Enhanced evaluation with iterative NNLS and local search"""
    led_indices, n_target = combination_data
    
    # Fit powers with iterative refinement
    powers = fit_powers_nnls_iterative(
        led_indices,
        optimizer_data['led_spectra'],
        optimizer_data['target_spectrum'],
        optimizer_data['wavelengths'],
        optimizer_data['weights'],
        optimizer_data['led_efficiencies'],
        optimizer_data['pmax'],
        optimizer_data['led_database']
    )
    
    # Remove LEDs with zero power
    active_mask = [p > 1e-6 for p in powers]
    if any(active_mask):
        led_indices = [idx for i, idx in enumerate(led_indices) if active_mask[i]]
        powers = [p for p in powers if p > 1e-6]
    
    # Calculate fitness with dynamic weights
    fitness, cls = calculate_fitness_with_dynamic_weights(
        led_indices, powers,
        optimizer_data['led_spectra'],
        optimizer_data['target_spectrum'],
        optimizer_data['wavelengths'],
        optimizer_data['weights'],
        n_target,
        optimizer_data.get('previous_cl_star'),
        optimizer_data.get('problematic_bins')
    )
    
    # Apply local search if enabled
    if optimizer_data.get('use_local_search', False):
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
# MAIN ENHANCED OPTIMIZER CLASS
# ============================================================================

class EnhancedParallelSolarSimulatorEA:
    """Enhanced LED Solar Simulator with Advanced Parallel EA"""
    
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
                 n_jobs: int = -1,
                 enable_led_clustering: bool = True,
                 enable_local_search: bool = True):
        
        self.led_database = led_database
        self.wavelength_range = wavelength_range
        self.spectral_resolution = spectral_resolution
        self.population_size = population_size
        self.max_leds = max_leds
        self.min_leds = min_leds
        self.max_total_power = max_total_power
        
        # Enhanced features
        self.enable_led_clustering = enable_led_clustering
        self.enable_local_search = enable_local_search
        
        # Geometry
        self.led_to_target_distance = led_to_target_distance
        self.target_area = target_area
        self.flux_to_irradiance = 1.0 / (np.pi * led_to_target_distance**2)
        
        # Parallel processing
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        print(f"🚀 Enhanced Parallel processing: {self.n_jobs} cores")
        
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
        self.base_weights = self.build_weights()
        
        print("🔄 Interpolating LED spectra...")
        self.interpolate_led_spectra()
        
        print("📐 Converting to irradiance units...")
        self.convert_leds_to_irradiance()
        
        print("📈 Calculating LED characteristics...")
        self.calculate_led_characteristics()
        
        # LED clustering
        if self.enable_led_clustering:
            print("🔗 Clustering similar LEDs...")
            self.create_led_clusters()
        
        self.pmax = 10.0
        self.fitness_history = []
        self.cl_star_history = []
        self.rls_history = []
        self.diversity_history = []
        self.multi_target_results = {}
        
        # Dynamic optimization state
        self.previous_best_cl_star = None
        self.problematic_bins = None
        
        print("✅ Enhanced Optimizer initialized!")
    
    def create_led_clusters(self, wavelength_tolerance: float = 10.0):
        """Create clusters of similar LEDs to prevent redundant selection"""
        if len(self.led_database) < 2:
            self.led_clusters = []
            return
        
        # Group LEDs by peak wavelength
        wavelength_groups = defaultdict(list)
        for i, led in enumerate(self.led_database):
            wl_bin = int(led.peak_wavelength / wavelength_tolerance) * wavelength_tolerance
            wavelength_groups[wl_bin].append(i)
        
        self.led_clusters = []
        cluster_id = 0
        
        for wl_bin, led_indices in wavelength_groups.items():
            if len(led_indices) > 1:
                # Choose representative LED (highest efficiency)
                representative = max(led_indices, key=lambda i: self.led_efficiencies[i])
                cluster = LEDCluster(led_indices, representative, wl_bin, cluster_id)
                self.led_clusters.append(cluster)
                cluster_id += 1
        
        print(f"   Created {len(self.led_clusters)} LED clusters")
    
    def get_cluster_representative(self, led_indices: List[int]) -> List[int]:
        """Replace clustered LEDs with their representatives"""
        if not hasattr(self, 'led_clusters') or not self.led_clusters:
            return led_indices
        
        # Create mapping from LED to cluster
        led_to_cluster = {}
        for cluster in self.led_clusters:
            for led_idx in cluster.led_indices:
                led_to_cluster[led_idx] = cluster
        
        # Replace with representatives, avoiding duplicates
        result = []
        used_clusters = set()
        
        for led_idx in led_indices:
            if led_idx in led_to_cluster:
                cluster = led_to_cluster[led_idx]
                if cluster.cluster_id not in used_clusters:
                    result.append(cluster.representative_idx)
                    used_clusters.add(cluster.cluster_id)
            else:
                result.append(led_idx)
        
        return result
    
    def calculate_population_diversity(self, population: List[LEDCombination]) -> float:
        """Calculate population diversity based on LED set differences"""
        if len(population) < 2:
            return 0.0
        
        total_diversity = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                set1 = set(population[i].led_indices)
                set2 = set(population[j].led_indices)
                
                # Jaccard distance
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                diversity = 1.0 - (intersection / union if union > 0 else 0.0)
                
                total_diversity += diversity
                comparisons += 1
        
        return total_diversity / comparisons if comparisons > 0 else 0.0
    
    def weighted_crossover(self, parent1: LEDCombination, parent2: LEDCombination, 
                          n_target: int) -> List[int]:
        """Enhanced crossover that weights selection based on parent fitness and LED power factors"""
        # Combine all LEDs from both parents
        combined_leds = list(set(parent1.led_indices + parent2.led_indices))
        
        if len(combined_leds) <= n_target:
            return combined_leds
        
        # Calculate selection weights based on parent fitness and LED power
        led_weights = {}
        
        # Weight from parent 1 (inverse fitness = better fitness gets higher weight)
        p1_weight = 1.0 / (parent1.fitness + 1e-6)
        for i, led_idx in enumerate(parent1.led_indices):
            if i < len(parent1.powers):
                power_weight = parent1.powers[i] + 1e-6
                led_weights[led_idx] = led_weights.get(led_idx, 0) + p1_weight * power_weight
        
        # Weight from parent 2
        p2_weight = 1.0 / (parent2.fitness + 1e-6)
        for i, led_idx in enumerate(parent2.led_indices):
            if i < len(parent2.powers):
                power_weight = parent2.powers[i] + 1e-6
                led_weights[led_idx] = led_weights.get(led_idx, 0) + p2_weight * power_weight
        
        # Normalize weights
        total_weight = sum(led_weights.values())
        if total_weight > 0:
            for led_idx in led_weights:
                led_weights[led_idx] /= total_weight
        
        # Select LEDs based on weights
        led_list = list(combined_leds)
        weights = [led_weights.get(led_idx, 1e-6) for led_idx in led_list]
        
        # Weighted selection without replacement
        selected = []
        remaining_leds = led_list.copy()
        remaining_weights = weights.copy()
        
        for _ in range(min(n_target, len(remaining_leds))):
            # Normalize current weights
            weight_sum = sum(remaining_weights)
            if weight_sum > 0:
                probs = [w / weight_sum for w in remaining_weights]
                choice_idx = np.random.choice(len(remaining_leds), p=probs)
            else:
                choice_idx = np.random.randint(len(remaining_leds))
            
            selected.append(remaining_leds[choice_idx])
            remaining_leds.pop(choice_idx)
            remaining_weights.pop(choice_idx)
        
        return selected
    
    def weakest_link_mutation(self, led_indices: List[int], powers: List[float], 
                             n_target: int) -> List[int]:
        """Enhanced mutation using weakest-link removal strategy"""
        if not led_indices or not powers:
            return led_indices
        
        mutation_type = np.random.choice(['remove_weakest', 'swap_weakest', 'add_strong'], 
                                       p=[0.4, 0.4, 0.2])
        
        # Identify weakest LEDs (lowest power factors)
        led_power_pairs = [(led_indices[i], powers[i] if i < len(powers) else 0.0) 
                          for i in range(len(led_indices))]
        led_power_pairs.sort(key=lambda x: x[1])  # Sort by power (ascending)
        
        if mutation_type == 'remove_weakest' and len(led_indices) > self.min_leds:
            # Remove the LED with lowest power factor
            weakest_led = led_power_pairs[0][0]
            return [led for led in led_indices if led != weakest_led]
        
        elif mutation_type == 'swap_weakest':
            # Replace weakest LED with a potentially better one
            if led_power_pairs:
                weakest_led = led_power_pairs[0][0]
                weakest_idx = led_indices.index(weakest_led)
                
                # Find available LEDs not in current selection
                available = [i for i in range(len(self.led_database)) if i not in led_indices]
                if available:
                    # Prefer LEDs with high efficiency in critical regions
                    critical_candidates = []
                    for led_idx in available:
                        peak = self.led_database[led_idx].peak_wavelength
                        if 600 <= peak <= 800:  # Critical region
                            critical_candidates.append(led_idx)
                    
                    if critical_candidates:
                        # Choose best efficiency from critical region
                        best_critical = max(critical_candidates, 
                                          key=lambda i: self.led_efficiencies[i])
                        new_indices = led_indices.copy()
                        new_indices[weakest_idx] = best_critical
                        return new_indices
                    else:
                        # Choose random available LED
                        new_indices = led_indices.copy()
                        new_indices[weakest_idx] = np.random.choice(available)
                        return new_indices
        
        elif mutation_type == 'add_strong' and len(led_indices) < n_target:
            # Add a strong LED from critical regions
            available = [i for i in range(len(self.led_database)) if i not in led_indices]
            if available:
                critical_candidates = []
                for led_idx in available:
                    peak = self.led_database[led_idx].peak_wavelength
                    if 600 <= peak <= 800:  # Critical region
                        critical_candidates.append(led_idx)
                
                if critical_candidates:
                    best_critical = max(critical_candidates, 
                                      key=lambda i: self.led_efficiencies[i])
                    return led_indices + [best_critical]
                else:
                    return led_indices + [np.random.choice(available)]
        
        return led_indices
    
    def enhanced_stagnation_detection(self, population: List[LEDCombination], 
                                    generation: int, no_improvement_count: int) -> bool:
        """Enhanced stagnation detection using population diversity"""
        # Calculate current population diversity
        diversity = self.calculate_population_diversity(population)
        self.diversity_history.append(diversity)
        
        # Stagnation criteria:
        # 1. No fitness improvement for many generations
        # 2. Low population diversity
        # 3. Combination of both
        
        fitness_stagnation = no_improvement_count > 20
        diversity_stagnation = diversity < 0.1  # Very low diversity
        
        # Look at diversity trend over last 10 generations
        if len(self.diversity_history) >= 10:
            recent_diversity = self.diversity_history[-10:]
            diversity_trend = np.mean(recent_diversity)
            diversity_declining = diversity_trend < 0.2
        else:
            diversity_declining = False
        
        # Trigger restart if multiple criteria met
        return (fitness_stagnation and diversity_stagnation) or \
               (no_improvement_count > 30) or \
               (diversity_declining and no_improvement_count > 15)
    
    def prepare_optimizer_data_enhanced(self, previous_best: LEDCombination = None) -> Dict:
        """Prepare enhanced data for parallel workers"""
        data = {
            'led_spectra': self.led_spectra,
            'target_spectrum': self.target_spectrum,
            'wavelengths': self.wavelengths,
            'weights': self.base_weights,
            'led_efficiencies': self.led_efficiencies,
            'pmax': self.pmax,
            'led_database': self.led_database,
            'use_local_search': self.enable_local_search
        }
        
        # Add dynamic weight information
        if previous_best is not None:
            combined = self.calculate_combined_spectrum(previous_best)
            sr_metrics = self.calculate_spectral_ratio_metrics(combined)
            data['previous_cl_star'] = sr_metrics['CL_star']
            
            # Identify problematic bins (CL > 0.2)
            problematic = [i for i, cl in enumerate(sr_metrics['CLs']) if cl > 0.2]
            data['problematic_bins'] = problematic
        
        return data
    
    def evolutionary_algorithm_enhanced(self, 
                                      n_target: int,
                                      pop_size: int = 50,
                                      n_generations: int = 100,
                                      elite_size: int = 5,
                                      tournament_size: int = 3,
                                      crossover_rate: float = 0.7,
                                      mutation_rate: float = 0.3) -> LEDCombination:
        """Enhanced Parallel EA with all improvements"""
        print(f"\n{'='*70}")
        print(f"🧬 ENHANCED PARALLEL EA: {n_target} LEDs | {self.n_jobs} cores")
        print(f"   Features: Clustering={self.enable_led_clustering}, LocalSearch={self.enable_local_search}")
        print(f"{'='*70}")
        
        # Initialize population
        population = self.initialize_population_enhanced(n_target, pop_size)
        best_ever = min(population, key=lambda x: x.fitness)
        
        # Initialize tracking
        self.fitness_history = []
        self.cl_star_history = []
        self.rls_history = []
        self.diversity_history = []
        
        # Initial metrics
        combined = self.calculate_combined_spectrum(best_ever)
        sr_metrics = self.calculate_spectral_ratio_metrics(combined)
        rls = self.calculate_relative_least_squares(combined)
        diversity = self.calculate_population_diversity(population)
        
        self.fitness_history.append(best_ever.fitness)
        self.cl_star_history.append(sr_metrics['CL_star'])
        self.rls_history.append(rls)
        self.diversity_history.append(diversity)
        
        print(f"\n📊 Initial: Fitness={best_ever.fitness:.6f} | CL*={sr_metrics['CL_star']:.4f} | RLS={rls:.4f} | Diversity={diversity:.3f}")
        print(f"\n{'Gen':<5} {'Fitness':<12} {'CL*':<10} {'RLS':<10} {'Div':<8} {'LEDs':<6} {'Status'}")
        print("-"*80)
        
        # Adaptive parameters
        no_improvement_count = 0
        current_mutation_rate = mutation_rate
        restart_count = 0
        
        for generation in range(n_generations):
            population.sort(key=lambda x: x.fitness)
            current_best = population[0]
            
            # Calculate metrics
            combined = self.calculate_combined_spectrum(current_best)
            sr_metrics = self.calculate_spectral_ratio_metrics(combined)
            rls = self.calculate_relative_least_squares(combined)
            diversity = self.calculate_population_diversity(population)
            
            # Update history
            self.fitness_history.append(current_best.fitness)
            self.cl_star_history.append(sr_metrics['CL_star'])
            self.rls_history.append(rls)
            self.diversity_history.append(diversity)
            
            # Check improvement
            improved = False
            if current_best.fitness < best_ever.fitness * 0.999:
                best_ever = current_best
                improved = True
                no_improvement_count = 0
                current_mutation_rate = mutation_rate
            else:
                no_improvement_count += 1
            
            # Adaptive mutation rate
            if no_improvement_count > 10:
                current_mutation_rate = min(0.6, mutation_rate * 1.5)
            elif no_improvement_count > 20:
                current_mutation_rate = min(0.8, mutation_rate * 2.0)
            
            # Enhanced stagnation detection
            if self.enhanced_stagnation_detection(population, generation, no_improvement_count):
                print(f"   🔄 ENHANCED RESTART: Stagnation detected (fitness + diversity). Restart #{restart_count + 1}")
                
                # Keep top 15%, replace rest with diverse solutions
                n_keep = max(3, pop_size // 7)
                population = population[:n_keep]
                
                # Generate diverse new individuals
                new_individuals = []
                for _ in range(pop_size - n_keep):
                    if np.random.random() < 0.3:
                        seed = self.generate_coverage_seed(n_target)
                    elif np.random.random() < 0.5:
                        seed = self.generate_region_balanced_seed(n_target)
                    else:
                        seed = self.generate_random_seed(n_target)
                    
                    # Apply clustering if enabled
                    if self.enable_led_clustering:
                        seed = self.get_cluster_representative(seed)
                    
                    new_individuals.append((seed, n_target))
                
                # Parallel evaluation with enhanced features
                optimizer_data = self.prepare_optimizer_data_enhanced(current_best)
                results = Parallel(n_jobs=self.n_jobs, backend='loky')(
                    delayed(evaluate_individual_enhanced)(ind, optimizer_data)
                    for ind in new_individuals
                )
                
                for led_indices, powers, fitness in results:
                    combo = LEDCombination(led_indices, powers, fitness, n_target)
                    population.append(combo)
                
                no_improvement_count = 0
                current_mutation_rate = mutation_rate * 1.2
                restart_count += 1
            
            # Print progress
            print_freq = 1 if n_generations <= 50 else 5
            if generation % print_freq == 0 or improved or generation == n_generations - 1:
                status = "★ NEW BEST" if improved else f"(stall: {no_improvement_count})" if no_improvement_count > 10 else ""
                
                cl_indicator = "✓✓" if sr_metrics['CL_star'] <= 0.05 else "✓ " if sr_metrics['CL_star'] <= 0.10 else "○ " if sr_metrics['CL_star'] <= 0.25 else "✗ "
                
                print(f"{generation:<5} {current_best.fitness:<12.6f} "
                      f"{sr_metrics['CL_star']:<10.4f} {rls:<10.4f} "
                      f"{diversity:<8.3f} {len(current_best.led_indices):<6} {cl_indicator} {status}")
            
            # Elitism
            next_population = population[:elite_size]
            
            # Generate offspring with enhanced methods
            n_offspring = pop_size - elite_size
            offspring_data = []
            
            # Prepare enhanced optimizer data
            optimizer_data = self.prepare_optimizer_data_enhanced(current_best)
            
            for _ in range(n_offspring):
                p1 = self.tournament_select(population, tournament_size)
                p2 = self.tournament_select(population, tournament_size)
                
                if np.random.random() < crossover_rate:
                    # Use weighted crossover
                    child = self.weighted_crossover(p1, p2, n_target)
                else:
                    child = p1.led_indices.copy()
                
                if np.random.random() < current_mutation_rate:
                    # Use weakest-link mutation
                    child = self.weakest_link_mutation(child, p1.powers, n_target)
                
                # Apply clustering if enabled
                if self.enable_led_clustering:
                    child = self.get_cluster_representative(child)
                
                offspring_data.append((child, n_target))
            
            # Parallel evaluation with enhanced features
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
        final_diversity = self.calculate_population_diversity(population)
        
        print(f"\n🏆 ENHANCED RESULTS:")
        print(f"   Fitness: {best_ever.fitness:.6f}")
        print(f"   CL*: {sr_metrics['CL_star']:.4f} (Class {sr_metrics['classification']})")
        print(f"   RLS: {rls:.4f}")
        print(f"   Final Diversity: {final_diversity:.3f}")
        print(f"   Restarts: {restart_count}")
        print(f"   Active LEDs: {len([p for p in best_ever.powers if p > 1e-6])}")
        
        # Show regional performance
        print(f"\n📊 Regional CL values:")
        regions = ['UV', 'Blue', 'Green', 'Red', 'NIR1', 'NIR2', 'NIR3', 'NIR4', 'NIR5']
        for i, (region, cl) in enumerate(zip(regions, sr_metrics['CLs'])):
            indicator = "✓" if cl <= 0.10 else "⚠️" if cl <= 0.25 else "✗"
            print(f"   {region:<8}: {cl:.4f} {indicator}")
        
        return best_ever
    
    def plot_pareto_front(self, results: Dict, save_path: str = None):
        """Plot Pareto front showing trade-off between LED count and CL*"""
        if not results or 'all_results' not in results:
            print("No multi-target results available for Pareto plot")
            return
        
        led_counts = []
        cl_stars = []
        classifications = []
        fitnesses = []
        
        for n_target, result in results['all_results'].items():
            led_counts.append(result['actual_led_count'])
            cl_stars.append(result['CL_star'])
            classifications.append(result['classification'])
            fitnesses.append(result['fitness'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: LED Count vs CL*
        colors = {'A+': 'green', 'A': 'orange', 'B': 'red', 'C': 'darkred'}
        for i, (count, cl_star, class_name) in enumerate(zip(led_counts, cl_stars, classifications)):
            # Check if label already exists
            existing_labels = [c.get_label() for c in ax1.collections if hasattr(c, 'get_label')]
            label = class_name if class_name not in existing_labels else None
            ax1.scatter(count, cl_star, c=colors.get(class_name, 'gray'), 
                       s=100, alpha=0.7, label=label)
        
        # Add classification boundaries
        ax1.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='A+ threshold')
        ax1.axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='Target threshold')
        ax1.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Class A threshold')
        
        ax1.set_xlabel('Number of LEDs')
        ax1.set_ylabel('CL* (Spectral Mismatch)')
        ax1.set_title('Pareto Front: LED Count vs Spectral Quality')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: LED Count vs Fitness
        for i, (count, fitness, class_name) in enumerate(zip(led_counts, fitnesses, classifications)):
            ax2.scatter(count, fitness, c=colors.get(class_name, 'gray'), 
                       s=100, alpha=0.7)
            ax2.annotate(f'{count}', (count, fitness), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Number of LEDs')
        ax2.set_ylabel('Fitness (lower is better)')
        ax2.set_title('LED Count vs Overall Fitness')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Pareto front saved: {save_path}")
        
        plt.show()
    
    def plot_enhanced_evolution(self, save_path: str = None):
        """Plot enhanced evolution history including diversity"""
        if not hasattr(self, 'diversity_history'):
            print("No enhanced evolution history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        generations = range(len(self.fitness_history))
        
        # Plot 1: Fitness evolution
        ax1 = axes[0, 0]
        ax1.plot(generations, self.fitness_history, 'b-', linewidth=2, label='Fitness')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness (lower is better)')
        ax1.set_title('Fitness Evolution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: CL* and RLS evolution
        ax2 = axes[0, 1]
        ax2.plot(generations, self.cl_star_history, 'r-', linewidth=2, label='CL*')
        ax2.plot(generations, self.rls_history, 'g-', linewidth=2, label='RLS')
        ax2.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='A+ (≤0.05)')
        ax2.axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='Target (≤0.10)')
        ax2.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Class A (≤0.25)')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Value')
        ax2.set_title('Spectral Quality Evolution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Population diversity
        ax3 = axes[1, 0]
        ax3.plot(generations, self.diversity_history, 'm-', linewidth=2, label='Diversity')
        ax3.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Low diversity threshold')
        ax3.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Moderate diversity')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Population Diversity')
        ax3.set_title('Population Diversity Evolution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Combined view (normalized)
        ax4 = axes[1, 1]
        # Normalize all metrics to [0, 1] for comparison
        norm_fitness = np.array(self.fitness_history) / max(self.fitness_history)
        norm_cl_star = np.array(self.cl_star_history) / max(self.cl_star_history)
        norm_diversity = np.array(self.diversity_history) / max(self.diversity_history) if max(self.diversity_history) > 0 else np.zeros_like(self.diversity_history)
        
        ax4.plot(generations, norm_fitness, 'b-', linewidth=2, label='Fitness (norm)')
        ax4.plot(generations, norm_cl_star, 'r-', linewidth=2, label='CL* (norm)')
        ax4.plot(generations, norm_diversity, 'm-', linewidth=2, label='Diversity (norm)')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Normalized Value')
        ax4.set_title('Combined Evolution (Normalized)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Enhanced evolution plot saved: {save_path}")
        
        plt.show()
    
    # Include all the base methods from the original class
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
    
    def calculate_combined_spectrum(self, combination: LEDCombination) -> np.ndarray:
        """Calculate combined spectrum"""
        combined = np.zeros_like(self.wavelengths)
        for led_idx, power in zip(combination.led_indices, combination.powers):
            if led_idx < len(self.led_database) and power > 1e-6:
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
        weighted_rls = rls_values * self.base_weights
        RLS = np.sum(weighted_rls) / np.sum(self.base_weights)
        return RLS
    
    # Add remaining methods from original class...
    def initialize_population_enhanced(self, n_target: int, pop_size: int) -> List:
        """Enhanced initialization with clustering support"""
        print(f"🔄 Enhanced initialization ({pop_size} individuals)...")
        
        candidates = []
        
        # Multiple greedy seeds
        for i in range(5):
            seed = self.generate_greedy_seed_variant(n_target, variant=i)
            if seed:
                if self.enable_led_clustering:
                    seed = self.get_cluster_representative(seed)
                candidates.append((seed, n_target))
        
        # Coverage-based seeds
        for _ in range(5):
            seed = self.generate_coverage_seed(n_target)
            if self.enable_led_clustering:
                seed = self.get_cluster_representative(seed)
            candidates.append((seed, n_target))
        
        # Region-balanced seeds
        n_balanced = int(pop_size * 0.35)
        for _ in range(n_balanced):
            seed = self.generate_region_balanced_seed(n_target)
            if self.enable_led_clustering:
                seed = self.get_cluster_representative(seed)
            candidates.append((seed, n_target))
        
        # Random seeds
        n_random = pop_size - len(candidates)
        for _ in range(n_random):
            seed = self.generate_random_seed(n_target)
            if self.enable_led_clustering:
                seed = self.get_cluster_representative(seed)
            candidates.append((seed, n_target))
        
        # Parallel evaluation with enhanced features
        optimizer_data = self.prepare_optimizer_data_enhanced()
        
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
    
    # Include all the seed generation methods from original class
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
        min_leds = max(self.min_leds, min(n_target - 3, 1))
        max_leds = min(n_target + 2, len(self.led_database))
        
        # Ensure valid range
        if min_leds >= max_leds:
            min_leds = min(max_leds - 1, 1)
        
        if min_leds >= max_leds:
            # Fallback: use target directly
            n_leds = min(n_target, len(self.led_database))
        else:
            n_leds = np.random.randint(min_leds, max_leds)
        
        return np.random.choice(
            len(self.led_database),
            size=n_leds,
            replace=False
        ).tolist()
    
    def tournament_select(self, population, size):
        """Tournament selection"""
        indices = np.random.choice(len(population), size=size, replace=False)
        tournament = [population[i] for i in indices]
        return min(tournament, key=lambda x: x.fitness)
    
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
    
    def optimize_multi_target_enhanced(self, 
                                     target_led_counts: List[int],
                                     sequential: bool = False) -> Dict:
        """Enhanced multi-target optimization with Pareto analysis"""
        print("\n" + "="*70)
        print("🎯 ENHANCED MULTI-TARGET PARALLEL OPTIMIZATION")
        print("="*70)
        print(f"Targets: {target_led_counts}")
        print(f"Mode: {'Sequential' if sequential else 'Parallel (ALL targets simultaneously)'}")
        print(f"Features: Clustering={self.enable_led_clustering}, LocalSearch={self.enable_local_search}")
        
        if sequential:
            results = self._optimize_sequential_enhanced(target_led_counts)
        else:
            results = self._optimize_truly_parallel_enhanced(target_led_counts)
        
        # Find best overall solution
        sorted_results = sorted(results.items(), 
                              key=lambda x: (x[1]['CL_star'], x[1]['fitness']))
        best_target = sorted_results[0][0]
        
        self.multi_target_results = {
            'all_results': results,
            'best_target': best_target,
            'best_solution': results[best_target]['solution']
        }
        
        # Print summary
        print(f"\n📊 MULTI-TARGET RESULTS SUMMARY:")
        print(f"{'Target':<8} {'Actual':<8} {'Fitness':<12} {'CL*':<10} {'Class':<6} {'RLS':<10}")
        print("-"*70)
        
        for n_target in sorted(target_led_counts):
            result = results[n_target]
            print(f"{n_target:<8} {result['actual_led_count']:<8} "
                  f"{result['fitness']:<12.6f} {result['CL_star']:<10.4f} "
                  f"{result['classification']:<6} {result['RLS']:<10.4f}")
        
        print(f"\n🏆 Best Overall: {best_target} LEDs (Class {results[best_target]['classification']})")
        
        return self.multi_target_results

    def _optimize_sequential_enhanced(self, target_led_counts):
        """Sequential optimization with enhanced features"""
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
            results[n_target] = self._package_result_enhanced(solution, n_target)
        return results

    def _optimize_truly_parallel_enhanced(self, target_led_counts):
        """Enhanced parallel optimization for all targets"""
        print(f"\n⚡ Running {len(target_led_counts)} targets SIMULTANEOUSLY with ENHANCED features")
        print(f"   Using ~{self.n_jobs // len(target_led_counts)} cores per target")
        print(f"   This should be ~{len(target_led_counts)}x faster!\n")
        
        def optimize_one_target_enhanced(n_target, cores_per_target):
            """Run enhanced optimization for one target"""
            # Create enhanced optimizer instance
            ea_instance = copy.deepcopy(self)
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
        target_results = Parallel(n_jobs=len(target_led_counts), backend='loky')(
            delayed(optimize_one_target_enhanced)(n_target, cores_per_target)
            for n_target in target_led_counts
        )
        
        # Package results
        results = {}
        for n_target, solution in target_results:
            results[n_target] = self._package_result_enhanced(solution, n_target)
        
        return results

    def _package_result_enhanced(self, solution, n_target):
        """Package solution with enhanced metrics"""
        combined = self.calculate_combined_spectrum(solution)
        sr_metrics = self.calculate_spectral_ratio_metrics(combined)
        RLS = self.calculate_relative_least_squares(combined)
        
        # Count only active LEDs
        active_leds = len([p for p in solution.powers if p > 1e-6])
        
        return {
            'solution': solution,
            'fitness': solution.fitness,
            'target_led_count': n_target,
            'actual_led_count': active_leds,
            'total_led_count': len(solution.led_indices),
            'RLS': RLS,
            'CL_star': sr_metrics['CL_star'],
            'classification': sr_metrics['classification'],
            'sr_metrics': sr_metrics,
            'active_power_sum': sum(p for p in solution.powers if p > 1e-6)
        }
    
    def save_enhanced_configuration(self, solution: LEDCombination, filename: str):
        """Save enhanced configuration with detailed metrics"""
        combined = self.calculate_combined_spectrum(solution)
        sr_metrics = self.calculate_spectral_ratio_metrics(combined)
        RLS = self.calculate_relative_least_squares(combined)
        
        # Count active LEDs
        active_leds = [(i, led_idx, power) for i, (led_idx, power) in 
                      enumerate(zip(solution.led_indices, solution.powers)) if power > 1e-6]
        
        config = {
            'optimization_summary': {
                'total_leds_in_solution': len(solution.led_indices),
                'active_leds': len(active_leds),
                'target_led_count': solution.n_target,
                'fitness': float(solution.fitness),
                'RLS': float(RLS),
                'CL_star': float(sr_metrics['CL_star']),
                'classification': sr_metrics['classification'],
                'total_active_power': float(sum(p for p in solution.powers if p > 1e-6))
            },
            'spectral_ratios': {
                f'SR_{i+1}': float(sr) for i, sr in enumerate(sr_metrics['spectral_ratios'])
            },
            'spectral_mismatches': {
                f'CL_{i+1}': float(cl) for i, cl in enumerate(sr_metrics['CLs'])
            },
            'led_configuration': [],
            'inactive_leds': []
        }
        
        # Active LEDs
        for i, led_idx, power in active_leds:
            led = self.led_database[led_idx]
            config['led_configuration'].append({
                'index': int(led_idx),
                'name': led.name,
                'peak_wavelength': float(led.peak_wavelength),
                'viewing_angle': float(led.viewing_angle),
                'power_factor': float(power),
                'manufacturer': getattr(led, 'manufacturer', 'Unknown'),
                'part_number': getattr(led, 'part_number', 'Unknown')
            })
        
        # Inactive LEDs (for reference)
        for i, (led_idx, power) in enumerate(zip(solution.led_indices, solution.powers)):
            if power <= 1e-6:
                led = self.led_database[led_idx]
                config['inactive_leds'].append({
                    'index': int(led_idx),
                    'name': led.name,
                    'peak_wavelength': float(led.peak_wavelength),
                    'power_factor': float(power)
                })
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Enhanced configuration saved: {filename}")
        print(f"   Active LEDs: {len(active_leds)}/{len(solution.led_indices)}")
        print(f"   Classification: {sr_metrics['classification']}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_sample_led_database(n_leds: int = 100) -> List[LEDSpec]:
    """Create a sample LED database for testing"""
    np.random.seed(42)  # For reproducibility
    
    leds = []
    
    # Define spectral regions with typical LED types
    led_types = [
        # (peak_wl, width, name_prefix, count)
        (365, 15, "UV", 8),
        (405, 20, "Violet", 10),
        (450, 25, "Blue", 15),
        (520, 30, "Green", 12),
        (590, 25, "Amber", 8),
        (630, 30, "Red", 15),
        (660, 25, "DeepRed", 12),
        (730, 35, "FarRed", 10),
        (850, 40, "NIR1", 8),
        (940, 45, "NIR2", 6),
        (1050, 50, "NIR3", 4),
        (1200, 60, "NIR4", 2)
    ]
    
    led_id = 0
    for peak_wl, width, name_prefix, count in led_types:
        for i in range(count):
            if led_id >= n_leds:
                break
            
            # Add some variation to peak wavelength
            actual_peak = peak_wl + np.random.normal(0, width * 0.1)
            
            # Create wavelength array
            wl_range = np.linspace(actual_peak - 3*width, actual_peak + 3*width, 100)
            
            # Create Gaussian spectrum
            intensities = np.exp(-(wl_range - actual_peak)**2 / (2 * (width/2.35)**2))
            
            # Add some noise and asymmetry
            intensities += np.random.normal(0, 0.02, len(intensities))
            intensities = np.maximum(intensities, 0)
            
            # Normalize
            intensities /= np.max(intensities)
            
            led = LEDSpec(
                name=f"{name_prefix}_{i+1}",
                peak_wavelength=actual_peak,
                viewing_angle=np.random.uniform(15, 120),
                wavelengths=wl_range,
                intensities=intensities,
                manufacturer=f"Manufacturer_{np.random.randint(1, 6)}",
                part_number=f"PN_{led_id:04d}"
            )
            
            leds.append(led)
            led_id += 1
    
    return leds[:n_leds]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("🌟 Enhanced LED Solar Simulator Optimizer")
    print("="*50)
    
    # Create sample LED database
    print("📦 Creating sample LED database...")
    led_database = create_sample_led_database(n_leds=80)
    print(f"   Created {len(led_database)} LEDs")
    
    # Initialize enhanced optimizer
    optimizer = EnhancedParallelSolarSimulatorEA(
        led_database=led_database,
        population_size=60,
        max_leds=35,
        min_leds=8,
        n_jobs=4,  # Adjust based on your system
        enable_led_clustering=True,
        enable_local_search=True
    )
    
    # Run multi-target optimization
    target_counts = [20, 25, 30]
    results = optimizer.optimize_multi_target_enhanced(
        target_led_counts=target_counts,
        sequential=False  # Use parallel mode
    )
    
    # Get best solution
    best_solution = results['best_solution']
    
    # Save configuration
    optimizer.save_enhanced_configuration(
        best_solution, 
        f"enhanced_led_config_{results['best_target']}_leds.json"
    )
    
    # Plot results
    optimizer.plot_enhanced_evolution("enhanced_evolution.png")
    optimizer.plot_pareto_front(results, "pareto_front.png")
    
    print("\n✅ Enhanced optimization complete!")