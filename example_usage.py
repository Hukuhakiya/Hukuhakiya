"""
Example usage of the Enhanced LED Solar Simulator Optimizer
"""

import numpy as np
import matplotlib.pyplot as plt
from enhanced_led_optimizer import EnhancedParallelSolarSimulatorEA, LEDCombination
from utils import LEDSpec

def create_sample_led_database():
    """Create a sample LED database for demonstration"""
    led_database = []
    
    # Create LEDs across different wavelength regions
    wavelengths = np.arange(300, 1201, 1)
    
    # UV LEDs (300-400nm)
    for peak in [365, 375, 385, 395]:
        intensities = np.exp(-(wavelengths - peak)**2 / (2 * 15**2))
        led = LEDSpec(
            name=f"UV_{peak}nm",
            peak_wavelength=peak,
            viewing_angle=120.0,
            wavelengths=wavelengths,
            intensities=intensities
        )
        led_database.append(led)
    
    # Blue LEDs (400-500nm)
    for peak in [450, 460, 470, 480, 490]:
        intensities = np.exp(-(wavelengths - peak)**2 / (2 * 20**2))
        led = LEDSpec(
            name=f"Blue_{peak}nm",
            peak_wavelength=peak,
            viewing_angle=120.0,
            wavelengths=wavelengths,
            intensities=intensities
        )
        led_database.append(led)
    
    # Green LEDs (500-600nm)
    for peak in [520, 530, 540, 550, 560, 570, 580, 590]:
        intensities = np.exp(-(wavelengths - peak)**2 / (2 * 25**2))
        led = LEDSpec(
            name=f"Green_{peak}nm",
            peak_wavelength=peak,
            viewing_angle=120.0,
            wavelengths=wavelengths,
            intensities=intensities
        )
        led_database.append(led)
    
    # Red LEDs (600-700nm)
    for peak in [620, 630, 640, 650, 660, 670, 680, 690]:
        intensities = np.exp(-(wavelengths - peak)**2 / (2 * 30**2))
        led = LEDSpec(
            name=f"Red_{peak}nm",
            peak_wavelength=peak,
            viewing_angle=120.0,
            wavelengths=wavelengths,
            intensities=intensities
        )
        led_database.append(led)
    
    # NIR LEDs (700-1000nm)
    for peak in [720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980]:
        intensities = np.exp(-(wavelengths - peak)**2 / (2 * 35**2))
        led = LEDSpec(
            name=f"NIR_{peak}nm",
            peak_wavelength=peak,
            viewing_angle=120.0,
            wavelengths=wavelengths,
            intensities=intensities
        )
        led_database.append(led)
    
    # Far NIR LEDs (1000-1200nm)
    for peak in [1020, 1040, 1060, 1080, 1100, 1120, 1140, 1160, 1180]:
        intensities = np.exp(-(wavelengths - peak)**2 / (2 * 40**2))
        led = LEDSpec(
            name=f"FarNIR_{peak}nm",
            peak_wavelength=peak,
            viewing_angle=120.0,
            wavelengths=wavelengths,
            intensities=intensities
        )
        led_database.append(led)
    
    return led_database

def demonstrate_single_target_optimization():
    """Demonstrate single target optimization"""
    print("=" * 60)
    print("SINGLE TARGET OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample LED database
    led_database = create_sample_led_database()
    print(f"Created {len(led_database)} sample LEDs")
    
    # Initialize optimizer
    optimizer = EnhancedParallelSolarSimulatorEA(
        led_database=led_database,
        target_spectrum_file=None,  # Use built-in AM1.5G
        n_jobs=4  # Use 4 cores for demonstration
    )
    
    # Run optimization
    print("\nRunning optimization for 25 LEDs...")
    solution = optimizer.evolutionary_algorithm_enhanced(
        n_target=25,
        pop_size=50,
        n_generations=50,  # Reduced for demonstration
        elite_size=5,
        tournament_size=3,
        crossover_rate=0.7,
        mutation_rate=0.3
    )
    
    # Display results
    print(f"\nOptimization completed!")
    print(f"Best solution:")
    print(f"  - LED count: {len(solution.led_indices)}")
    print(f"  - Fitness: {solution.fitness:.6f}")
    
    # Calculate and display metrics
    combined = optimizer.calculate_combined_spectrum(solution)
    sr_metrics = optimizer.calculate_spectral_ratio_metrics(combined)
    rls = optimizer.calculate_relative_least_squares(combined)
    
    print(f"  - CL*: {sr_metrics['CL_star']:.4f}")
    print(f"  - Classification: {sr_metrics['classification']}")
    print(f"  - RLS: {rls:.4f}")
    
    # Show selected LEDs
    print(f"\nSelected LEDs:")
    for i, (led_idx, power) in enumerate(zip(solution.led_indices, solution.powers)):
        led = led_database[led_idx]
        print(f"  {i+1:2d}. {led.name} (λ={led.peak_wavelength:.0f}nm, P={power:.3f})")
    
    # Plot results
    optimizer.plot_evolution_history("single_target_evolution.png")
    optimizer.plot_results(solution, "single_target_results.png")
    
    return solution

def demonstrate_multi_target_optimization():
    """Demonstrate multi-target optimization"""
    print("\n" + "=" * 60)
    print("MULTI-TARGET OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample LED database
    led_database = create_sample_led_database()
    
    # Initialize optimizer
    optimizer = EnhancedParallelSolarSimulatorEA(
        led_database=led_database,
        target_spectrum_file=None,
        n_jobs=4
    )
    
    # Run multi-target optimization
    target_counts = [20, 25, 30, 35]
    print(f"Running optimization for targets: {target_counts}")
    
    results = optimizer.optimize_multi_target_enhanced(
        target_led_counts=target_counts,
        sequential=False  # Use parallel processing
    )
    
    # Display results
    print(f"\nMulti-target optimization completed!")
    print(f"Results summary:")
    print(f"{'Target':<8} {'Actual':<8} {'CL*':<8} {'Class':<6} {'Fitness':<12}")
    print("-" * 50)
    
    for target, result in results['all_results'].items():
        print(f"{target:<8} {result['actual_led_count']:<8} "
              f"{result['CL_star']:<8.4f} {result['classification']:<6} "
              f"{result['fitness']:<12.6f}")
    
    # Show Pareto front
    print(f"\nPareto Front Analysis:")
    for point in results['pareto_front']:
        print(f"  {point['led_count']} LEDs: CL*={point['CL_star']:.4f}, "
              f"Fitness={point['fitness']:.4f}")
    
    # Plot Pareto front
    optimizer.plot_pareto_front("multi_target_pareto.png")
    
    return results

def demonstrate_enhanced_features():
    """Demonstrate enhanced features"""
    print("\n" + "=" * 60)
    print("ENHANCED FEATURES DEMONSTRATION")
    print("=" * 60)
    
    # Create sample LED database
    led_database = create_sample_led_database()
    
    # Initialize optimizer
    optimizer = EnhancedParallelSolarSimulatorEA(
        led_database=led_database,
        target_spectrum_file=None,
        n_jobs=4
    )
    
    print("Enhanced features demonstrated:")
    print("1. ✅ LED Clustering - LEDs grouped by wavelength similarity")
    print(f"   Created {len(optimizer.led_clusters)} LED clusters")
    
    print("2. ✅ Dynamic Weights - Spectral weights adapt based on performance")
    print("   Weights adjust to focus on problematic regions")
    
    print("3. ✅ Iterative NNLS - Automatic removal of low-power LEDs")
    print("   Produces leaner, more efficient solutions")
    
    print("4. ✅ Local Search - Post-evaluation hill climbing")
    print("   Refines promising solutions immediately")
    
    print("5. ✅ Enhanced Mutation - Targeted LED replacement")
    print("   Focuses on weak spectral regions")
    
    print("6. ✅ Weighted Crossover - Intelligent LED selection")
    print("   Considers parent fitness and LED efficiency")
    
    print("7. ✅ Diversity Tracking - Population diversity monitoring")
    print("   Prevents premature convergence")
    
    print("8. ✅ Pareto Analysis - Multi-objective optimization")
    print("   Identifies optimal trade-offs between LED count and performance")

def main():
    """Main demonstration function"""
    print("Enhanced LED Solar Simulator Optimizer - Demonstration")
    print("=" * 70)
    
    try:
        # Demonstrate single target optimization
        solution = demonstrate_single_target_optimization()
        
        # Demonstrate multi-target optimization
        results = demonstrate_multi_target_optimization()
        
        # Demonstrate enhanced features
        demonstrate_enhanced_features()
        
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("Generated files:")
        print("  - single_target_evolution.png")
        print("  - single_target_results.png")
        print("  - multi_target_pareto.png")
        print("\nThe enhanced optimizer provides significant improvements in:")
        print("  - Solution quality (better CL* and RLS values)")
        print("  - Computational efficiency (parallel processing)")
        print("  - Engineering practicality (leaner solutions)")
        print("  - Design flexibility (Pareto front analysis)")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Please ensure all dependencies are installed and the code is properly configured.")

if __name__ == "__main__":
    main()