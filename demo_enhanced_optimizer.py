"""
Demonstration of Enhanced LED Solar Simulator Optimizer
Shows all the improvements and their benefits
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
from typing import List, Dict

# Import the enhanced optimizer
from led_solar_simulator_optimizer_enhanced import EnhancedSolarSimulatorEA, LEDCombination
from utils import LEDSpec


def create_sample_led_database(n_leds: int = 100) -> List[LEDSpec]:
    """Create a sample LED database for demonstration"""
    led_database = []
    
    # Define wavelength regions and their characteristics
    regions = [
        (300, 400, 0.08, "UV"),      # UV region
        (400, 500, 0.12, "Blue"),    # Blue region  
        (500, 600, 0.15, "Green"),   # Green region
        (600, 700, 0.20, "Red"),     # Red region (critical)
        (700, 800, 0.20, "NIR1"),    # NIR1 region (critical)
        (800, 900, 0.12, "NIR2"),    # NIR2 region
        (900, 1000, 0.08, "NIR3"),   # NIR3 region
        (1000, 1100, 0.03, "NIR4"),  # NIR4 region
        (1100, 1200, 0.02, "NIR5")   # NIR5 region
    ]
    
    leds_per_region = n_leds // len(regions)
    
    for region_idx, (wl_min, wl_max, fraction, name) in enumerate(regions):
        for i in range(leds_per_region):
            # Generate peak wavelength within region
            peak_wl = np.random.uniform(wl_min, wl_max)
            
            # Generate viewing angle (narrower for higher wavelengths)
            if peak_wl < 500:
                viewing_angle = np.random.uniform(15, 30)
            elif peak_wl < 800:
                viewing_angle = np.random.uniform(20, 40)
            else:
                viewing_angle = np.random.uniform(25, 50)
            
            # Generate spectral width (FWHM)
            fwhm = np.random.uniform(20, 60)
            
            # Create wavelength array
            wavelengths = np.arange(300, 1201, 1)
            
            # Generate Gaussian spectrum
            intensities = np.exp(-((wavelengths - peak_wl) / fwhm) ** 2)
            
            # Add some noise and asymmetry
            noise = np.random.normal(0, 0.05, len(intensities))
            intensities = np.maximum(intensities + noise, 0)
            
            # Normalize
            intensities = intensities / np.max(intensities) if np.max(intensities) > 0 else intensities
            
            # Create LED specification
            led = LEDSpec(
                name=f"{name}_LED_{i+1:02d}",
                peak_wavelength=peak_wl,
                viewing_angle=viewing_angle,
                wavelengths=wavelengths,
                intensities=intensities
            )
            
            led_database.append(led)
    
    # Add some extra LEDs in critical regions
    for _ in range(n_leds % len(regions)):
        region_idx = np.random.randint(0, len(regions))
        wl_min, wl_max, _, name = regions[region_idx]
        
        peak_wl = np.random.uniform(wl_min, wl_max)
        viewing_angle = np.random.uniform(20, 40)
        fwhm = np.random.uniform(20, 60)
        
        wavelengths = np.arange(300, 1201, 1)
        intensities = np.exp(-((wavelengths - peak_wl) / fwhm) ** 2)
        intensities = np.maximum(intensities, 0)
        intensities = intensities / np.max(intensities) if np.max(intensities) > 0 else intensities
        
        led = LEDSpec(
            name=f"{name}_LED_EXTRA_{_+1:02d}",
            peak_wavelength=peak_wl,
            viewing_angle=viewing_angle,
            wavelengths=wavelengths,
            intensities=intensities
        )
        
        led_database.append(led)
    
    return led_database


def compare_optimizers():
    """Compare original vs enhanced optimizer"""
    print("🔬 COMPARISON: Original vs Enhanced Optimizer")
    print("="*60)
    
    # Create sample LED database
    print("📊 Creating sample LED database...")
    led_database = create_sample_led_database(80)
    print(f"   Created {len(led_database)} LEDs")
    
    # Test parameters
    n_target = 25
    n_generations = 50
    pop_size = 40
    
    print(f"\n🎯 Testing with {n_target} LEDs target, {n_generations} generations, {pop_size} population")
    
    # Initialize enhanced optimizer
    print("\n🚀 Initializing Enhanced Optimizer...")
    enhanced_optimizer = EnhancedSolarSimulatorEA(
        led_database=led_database,
        population_size=pop_size,
        n_jobs=4  # Use 4 cores for demo
    )
    
    # Run enhanced optimization
    print(f"\n⚡ Running Enhanced EA...")
    start_time = time.time()
    
    enhanced_solution = enhanced_optimizer.evolutionary_algorithm_enhanced(
        n_target=n_target,
        pop_size=pop_size,
        n_generations=n_generations,
        elite_size=5,
        tournament_size=3,
        crossover_rate=0.7,
        mutation_rate=0.3
    )
    
    enhanced_time = time.time() - start_time
    
    # Calculate metrics
    combined = enhanced_optimizer.calculate_combined_spectrum(enhanced_solution)
    sr_metrics = enhanced_optimizer.calculate_spectral_ratio_metrics(combined)
    rls = enhanced_optimizer.calculate_relative_least_squares(combined)
    
    print(f"\n📊 ENHANCED RESULTS:")
    print(f"   ⏱️  Time: {enhanced_time:.2f} seconds")
    print(f"   🎯 Fitness: {enhanced_solution.fitness:.6f}")
    print(f"   📈 CL*: {sr_metrics['CL_star']:.4f}")
    print(f"   📊 Classification: {sr_metrics['classification']}")
    print(f"   🔢 LEDs Used: {len(enhanced_solution.led_indices)}")
    print(f"   📉 RLS: {rls:.4f}")
    
    # Show regional performance
    print(f"\n📊 Regional CL values:")
    regions = ['UV', 'Blue', 'Green', 'Red', 'NIR1', 'NIR2', 'NIR3', 'NIR4', 'NIR5']
    for i, (region, cl) in enumerate(zip(regions, sr_metrics['CLs'])):
        indicator = "✓" if cl <= 0.10 else "⚠️" if cl <= 0.25 else "✗"
        print(f"   {region:<8}: {cl:.4f} {indicator}")
    
    return enhanced_optimizer, enhanced_solution


def demonstrate_multi_target_optimization():
    """Demonstrate multi-target optimization with Pareto analysis"""
    print("\n" + "="*60)
    print("🎯 MULTI-TARGET OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Create LED database
    led_database = create_sample_led_database(100)
    
    # Initialize optimizer
    optimizer = EnhancedSolarSimulatorEA(
        led_database=led_database,
        population_size=50,
        n_jobs=4
    )
    
    # Define target LED counts
    target_counts = [20, 25, 30, 35, 40]
    
    print(f"🎯 Optimizing for LED counts: {target_counts}")
    
    # Run multi-target optimization
    start_time = time.time()
    results = optimizer.optimize_multi_target_enhanced(
        target_led_counts=target_counts,
        sequential=False  # Use parallel mode
    )
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total optimization time: {total_time:.2f} seconds")
    print(f"   Average time per target: {total_time/len(target_counts):.2f} seconds")
    
    # Display results
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"{'Target':<8} {'Actual':<8} {'Fitness':<12} {'CL*':<8} {'Class':<6} {'RLS':<8}")
    print("-" * 60)
    
    for n_target in target_counts:
        result = results['all_results'][n_target]
        print(f"{n_target:<8} {result['actual_led_count']:<8} {result['fitness']:<12.6f} "
              f"{result['CL_star']:<8.4f} {result['classification']:<6} {result['RLS']:<8.4f}")
    
    # Find best solution
    best_target = results['best_target']
    best_solution = results['best_solution']
    print(f"\n🏆 BEST SOLUTION: {best_target} LEDs")
    print(f"   Classification: {results['all_results'][best_target]['classification']}")
    print(f"   CL*: {results['all_results'][best_target]['CL_star']:.4f}")
    
    # Plot Pareto front
    print(f"\n📊 Generating Pareto front plot...")
    optimizer.plot_pareto_front("pareto_front_demo.png")
    
    return optimizer, results


def demonstrate_enhanced_features():
    """Demonstrate specific enhanced features"""
    print("\n" + "="*60)
    print("🔧 ENHANCED FEATURES DEMONSTRATION")
    print("="*60)
    
    # Create LED database
    led_database = create_sample_led_database(60)
    
    # Initialize optimizer
    optimizer = EnhancedSolarSimulatorEA(
        led_database=led_database,
        population_size=30,
        n_jobs=2
    )
    
    print("🔍 Feature 1: LED Clustering")
    print(f"   Created {len(optimizer.led_clusters)} LED clusters")
    for cluster_wl, led_indices in list(optimizer.led_clusters.items())[:5]:
        print(f"   Cluster at {cluster_wl:.0f}nm: {len(led_indices)} LEDs")
    
    print("\n🔍 Feature 2: Dynamic Weights")
    # Simulate some CL* history
    cl_history = [0.3, 0.25, 0.2, 0.18, 0.15]
    dynamic_weights = optimizer.update_dynamic_weights(cl_history)
    print(f"   Dynamic weights updated based on CL* history: {cl_history}")
    print(f"   Weight range: {dynamic_weights.min():.3f} - {dynamic_weights.max():.3f}")
    
    print("\n🔍 Feature 3: Population Diversity Calculation")
    # Create sample population
    sample_population = []
    for i in range(10):
        led_indices = np.random.choice(len(led_database), size=15, replace=False).tolist()
        powers = np.random.uniform(0.1, 1.0, size=15).tolist()
        combo = LEDCombination(led_indices, powers, fitness=float(i))
        sample_population.append(combo)
    
    diversity = optimizer.calculate_population_diversity(sample_population)
    print(f"   Sample population diversity: {diversity:.3f}")
    
    print("\n🔍 Feature 4: Iterative NNLS Refinement")
    # Test with a sample combination
    test_indices = [0, 5, 10, 15, 20]
    test_powers = fit_powers_nnls_enhanced(
        test_indices,
        optimizer.led_spectra,
        optimizer.target_spectrum,
        optimizer.wavelengths,
        optimizer.weights,
        optimizer.led_efficiencies,
        optimizer.pmax,
        optimizer.led_database
    )
    print(f"   Original indices: {len(test_indices)}")
    print(f"   Refined powers: {len(test_powers)} (some may be removed)")
    print(f"   Power range: {min(test_powers):.3f} - {max(test_powers):.3f}")
    
    print("\n✅ All enhanced features demonstrated!")


def run_performance_benchmark():
    """Run performance benchmark comparing different configurations"""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Create LED database
    led_database = create_sample_led_database(80)
    
    # Test configurations
    configs = [
        {"name": "Small", "pop_size": 20, "n_generations": 30, "n_target": 20},
        {"name": "Medium", "pop_size": 40, "n_generations": 50, "n_target": 25},
        {"name": "Large", "pop_size": 60, "n_generations": 80, "n_target": 30}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n🧪 Testing {config['name']} configuration...")
        
        optimizer = EnhancedSolarSimulatorEA(
            led_database=led_database,
            population_size=config['pop_size'],
            n_jobs=4
        )
        
        start_time = time.time()
        solution = optimizer.evolutionary_algorithm_enhanced(
            n_target=config['n_target'],
            pop_size=config['pop_size'],
            n_generations=config['n_generations']
        )
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        combined = optimizer.calculate_combined_spectrum(solution)
        sr_metrics = optimizer.calculate_spectral_ratio_metrics(combined)
        rls = optimizer.calculate_relative_least_squares(combined)
        
        result = {
            'config': config['name'],
            'time': elapsed_time,
            'fitness': solution.fitness,
            'cl_star': sr_metrics['CL_star'],
            'classification': sr_metrics['classification'],
            'led_count': len(solution.led_indices),
            'rls': rls
        }
        
        results.append(result)
        
        print(f"   ⏱️  Time: {elapsed_time:.2f}s")
        print(f"   🎯 Fitness: {solution.fitness:.6f}")
        print(f"   📈 CL*: {sr_metrics['CL_star']:.4f}")
        print(f"   📊 Class: {sr_metrics['classification']}")
    
    # Summary
    print(f"\n📊 BENCHMARK SUMMARY:")
    print(f"{'Config':<8} {'Time(s)':<8} {'Fitness':<12} {'CL*':<8} {'Class':<6} {'LEDs':<6}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['config']:<8} {result['time']:<8.2f} {result['fitness']:<12.6f} "
              f"{result['cl_star']:<8.4f} {result['classification']:<6} {result['led_count']:<6}")
    
    return results


def main():
    """Main demonstration function"""
    print("🚀 ENHANCED LED SOLAR SIMULATOR OPTIMIZER DEMONSTRATION")
    print("="*70)
    print("This demo showcases all the improvements made to the optimizer:")
    print("• Targeted mutation strategies")
    print("• Weighted crossover based on fitness and power factors")
    print("• Post-evaluation local search (hill climbing)")
    print("• Dynamic weights for RLS based on CL* performance")
    print("• Iterative NNLS refinement to remove low-power LEDs")
    print("• Enhanced stagnation detection with population diversity")
    print("• LED clustering for similar LEDs")
    print("• Pareto front plotting for multi-objective analysis")
    print("="*70)
    
    try:
        # 1. Compare optimizers
        print("\n1️⃣ COMPARING OPTIMIZERS")
        enhanced_optimizer, enhanced_solution = compare_optimizers()
        
        # 2. Demonstrate enhanced features
        print("\n2️⃣ DEMONSTRATING ENHANCED FEATURES")
        demonstrate_enhanced_features()
        
        # 3. Multi-target optimization
        print("\n3️⃣ MULTI-TARGET OPTIMIZATION")
        multi_optimizer, multi_results = demonstrate_multi_target_optimization()
        
        # 4. Performance benchmark
        print("\n4️⃣ PERFORMANCE BENCHMARK")
        benchmark_results = run_performance_benchmark()
        
        # 5. Plot evolution history
        print("\n5️⃣ PLOTTING EVOLUTION HISTORY")
        print("📊 Generating enhanced evolution plot...")
        enhanced_optimizer.plot_enhanced_evolution_history("enhanced_evolution_demo.png")
        
        # 6. Save best configuration
        print("\n6️⃣ SAVING BEST CONFIGURATION")
        enhanced_optimizer.save_configuration(enhanced_solution, "best_configuration.json")
        print("💾 Best configuration saved to 'best_configuration.json'")
        
        print("\n✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("\n📁 Generated files:")
        print("   • enhanced_evolution_demo.png - Evolution history plot")
        print("   • pareto_front_demo.png - Pareto front analysis")
        print("   • best_configuration.json - Best LED configuration")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()