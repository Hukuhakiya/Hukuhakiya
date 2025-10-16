"""
Test script for the Enhanced LED Solar Simulator Optimizer
Verifies all improvements are working correctly
"""

import numpy as np
import sys
import traceback
from enhanced_led_optimizer import EnhancedParallelSolarSimulatorEA, create_sample_led_database


def test_basic_functionality():
    """Test basic optimizer functionality"""
    print("🧪 Testing Basic Functionality...")
    
    try:
        # Create small LED database for testing
        led_database = create_sample_led_database(n_leds=20)
        
        # Initialize optimizer
        optimizer = EnhancedParallelSolarSimulatorEA(
            led_database=led_database,
            population_size=10,
            max_leds=15,
            min_leds=5,
            n_jobs=2,
            enable_led_clustering=True,
            enable_local_search=True
        )
        
        print("   ✅ Optimizer initialization successful")
        
        # Test single optimization
        solution = optimizer.evolutionary_algorithm_enhanced(
            n_target=10,
            pop_size=10,
            n_generations=5,
            elite_size=2
        )
        
        print(f"   ✅ Single optimization successful (fitness: {solution.fitness:.6f})")
        
        # Test spectrum calculation
        combined = optimizer.calculate_combined_spectrum(solution)
        sr_metrics = optimizer.calculate_spectral_ratio_metrics(combined)
        
        print(f"   ✅ Spectrum analysis successful (CL*: {sr_metrics['CL_star']:.4f})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_enhanced_features():
    """Test enhanced features specifically"""
    print("\n🔬 Testing Enhanced Features...")
    
    try:
        led_database = create_sample_led_database(n_leds=30)
        
        optimizer = EnhancedParallelSolarSimulatorEA(
            led_database=led_database,
            population_size=15,
            n_jobs=2,
            enable_led_clustering=True,
            enable_local_search=True
        )
        
        # Test LED clustering
        if hasattr(optimizer, 'led_clusters'):
            print(f"   ✅ LED clustering: {len(optimizer.led_clusters)} clusters created")
        
        # Test enhanced initialization
        population = optimizer.initialize_population_enhanced(n_target=10, pop_size=15)
        print(f"   ✅ Enhanced initialization: {len(population)} individuals created")
        
        # Test population diversity calculation
        diversity = optimizer.calculate_population_diversity(population)
        print(f"   ✅ Population diversity calculation: {diversity:.3f}")
        
        # Test weighted crossover
        if len(population) >= 2:
            child = optimizer.weighted_crossover(population[0], population[1], n_target=10)
            print(f"   ✅ Weighted crossover: {len(child)} LEDs selected")
        
        # Test weakest link mutation
        if population[0].powers:
            mutated = optimizer.weakest_link_mutation(
                population[0].led_indices.copy(), 
                population[0].powers.copy(), 
                n_target=10
            )
            print(f"   ✅ Weakest link mutation: {len(mutated)} LEDs after mutation")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced features test failed: {e}")
        traceback.print_exc()
        return False


def test_multi_target_optimization():
    """Test multi-target parallel optimization"""
    print("\n🎯 Testing Multi-Target Optimization...")
    
    try:
        led_database = create_sample_led_database(n_leds=25)
        
        optimizer = EnhancedParallelSolarSimulatorEA(
            led_database=led_database,
            population_size=10,
            n_jobs=2,
            enable_led_clustering=True,
            enable_local_search=False  # Disable for faster testing
        )
        
        # Test multi-target optimization
        results = optimizer.optimize_multi_target_enhanced(
            target_led_counts=[8, 12],
            sequential=False
        )
        
        print(f"   ✅ Multi-target optimization successful")
        print(f"   📊 Results for {len(results['all_results'])} targets")
        
        for target, result in results['all_results'].items():
            print(f"      Target {target}: {result['actual_led_count']} LEDs, "
                  f"CL*={result['CL_star']:.4f}, Class {result['classification']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Multi-target optimization test failed: {e}")
        traceback.print_exc()
        return False


def test_configuration_export():
    """Test configuration export functionality"""
    print("\n💾 Testing Configuration Export...")
    
    try:
        led_database = create_sample_led_database(n_leds=15)
        
        optimizer = EnhancedParallelSolarSimulatorEA(
            led_database=led_database,
            population_size=8,
            n_jobs=2
        )
        
        # Quick optimization
        solution = optimizer.evolutionary_algorithm_enhanced(
            n_target=8,
            pop_size=8,
            n_generations=3
        )
        
        # Test configuration export
        config_file = "test_config.json"
        optimizer.save_enhanced_configuration(solution, config_file)
        print(f"   ✅ Configuration exported to {config_file}")
        
        # Verify file exists and is valid JSON
        import json
        import os
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            print(f"   ✅ Configuration file valid JSON")
            print(f"   📊 Active LEDs: {config['optimization_summary']['active_leds']}")
            print(f"   🏅 Classification: {config['optimization_summary']['classification']}")
            
            # Clean up
            os.remove(config_file)
            print(f"   🧹 Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration export test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("🚀 Enhanced LED Solar Simulator Optimizer - Test Suite")
    print("="*60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Enhanced Features", test_enhanced_features),
        ("Multi-Target Optimization", test_multi_target_optimization),
        ("Configuration Export", test_configuration_export)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "="*60)
    print("🏁 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name:<25}: {status}")
        if success:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED! Enhanced optimizer is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)