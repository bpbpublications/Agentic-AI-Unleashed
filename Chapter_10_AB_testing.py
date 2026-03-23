"""
A/B Testing framework for comparing two versions of support agent classifiers.
Tests challenger (A) vs incumbent (B) on latency and accuracy metrics with 
statistical significance testing to determine deployment readiness.
"""

import time
import random
import statistics
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import numpy as np


@dataclass
class ClassificationResult:
    """Result of a single classification request."""
    category: str
    confidence: float
    latency_ms: float
    correct: bool


class MockSupportAgent:
    """Mock support agent for A/B testing demonstration."""
    
    def __init__(self, name: str, base_latency: float, accuracy_rate: float):
        self.name = name
        self.base_latency = base_latency  # Base latency in milliseconds
        self.accuracy_rate = accuracy_rate
        self.categories = ["BILLING", "TECHNICAL", "ACCOUNT", "FEATURE", "GENERAL"]
    
    def classify_request(self, request: str, expected_category: str) -> ClassificationResult:
        """Classify a support request with simulated latency and accuracy."""
        # Simulate processing time with some variance
        start_time = time.time()
        latency = self.base_latency + random.gauss(0, self.base_latency * 0.2)
        time.sleep(latency / 1000)  # Convert to seconds for sleep
        
        # Simulate classification accuracy
        is_correct = random.random() < self.accuracy_rate
        predicted_category = expected_category if is_correct else random.choice(self.categories)
        
        actual_latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        confidence = random.uniform(0.7, 0.95) if is_correct else random.uniform(0.3, 0.7)
        
        return ClassificationResult(
            category=predicted_category,
            confidence=confidence,
            latency_ms=actual_latency,
            correct=is_correct
        )


class ABTestFramework:
    """A/B testing framework for support agent comparison."""
    
    def __init__(self, challenger: MockSupportAgent, incumbent: MockSupportAgent):
        self.challenger = challenger
        self.incumbent = incumbent
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self) -> List[Tuple[str, str]]:
        """Generate test dataset with request-category pairs."""
        test_cases = [
            ("My credit card was charged twice", "BILLING"),
            ("App crashes when uploading files", "TECHNICAL"),
            ("Can't reset my password", "ACCOUNT"),
            ("Need dark mode feature", "FEATURE"),
            ("What are your business hours?", "GENERAL"),
            ("Invoice download not working", "BILLING"),
            ("Login page shows error 500", "TECHNICAL"),
            ("How to change email address?", "ACCOUNT"),
            ("Can you add export functionality?", "FEATURE"),
            ("Where is your office located?", "GENERAL"),
        ]
        # Replicate test cases to get larger sample size
        return test_cases * 10
    
    def run_test(self, sample_size: int = None) -> Dict[str, List[ClassificationResult]]:
        """Run A/B test and collect results."""
        if sample_size is None:
            sample_size = len(self.test_data)
        
        results = {"challenger": [], "incumbent": []}
        test_sample = random.sample(self.test_data, min(sample_size, len(self.test_data)))
        
        print(f"Running A/B test with {len(test_sample)} samples per variant...")
        
        for request, expected_category in test_sample:
            # Test challenger (A)
            challenger_result = self.challenger.classify_request(request, expected_category)
            results["challenger"].append(challenger_result)
            
            # Test incumbent (B)
            incumbent_result = self.incumbent.classify_request(request, expected_category)
            results["incumbent"].append(incumbent_result)
        
        return results
    
    def calculate_metrics(self, results: List[ClassificationResult]) -> Dict[str, float]:
        """Calculate performance metrics from test results."""
        latencies = [r.latency_ms for r in results]
        accuracies = [r.correct for r in results]
        
        return {
            "mean_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "accuracy_rate": statistics.mean(accuracies),
            "total_requests": len(results),
            "latency_std": statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
    
    def statistical_significance(self, challenger_data: List[float], 
                               incumbent_data: List[float]) -> Tuple[float, bool]:
        """Perform t-test for statistical significance."""
        t_stat, p_value = stats.ttest_ind(challenger_data, incumbent_data)
        is_significant = p_value < 0.05
        return p_value, is_significant
    
    def print_scorecard(self, results: Dict[str, List[ClassificationResult]]):
        """Print comprehensive A/B test scorecard."""
        challenger_metrics = self.calculate_metrics(results["challenger"])
        incumbent_metrics = self.calculate_metrics(results["incumbent"])
        
        # Statistical significance tests
        challenger_latencies = [r.latency_ms for r in results["challenger"]]
        incumbent_latencies = [r.latency_ms for r in results["incumbent"]]
        challenger_accuracies = [float(r.correct) for r in results["challenger"]]
        incumbent_accuracies = [float(r.correct) for r in results["incumbent"]]
        
        latency_p_value, latency_sig = self.statistical_significance(
            challenger_latencies, incumbent_latencies)
        accuracy_p_value, accuracy_sig = self.statistical_significance(
            challenger_accuracies, incumbent_accuracies)
        
        # Calculate improvements
        latency_improvement = ((incumbent_metrics["mean_latency_ms"] - 
                              challenger_metrics["mean_latency_ms"]) / 
                             incumbent_metrics["mean_latency_ms"] * 100)
        accuracy_improvement = ((challenger_metrics["accuracy_rate"] - 
                               incumbent_metrics["accuracy_rate"]) * 100)
        
        print("\n" + "="*70)
        print("🔬 A/B TEST SCORECARD - SUPPORT AGENT CLASSIFICATION")
        print("="*70)
        
        print(f"\n📊 SAMPLE SIZE: {challenger_metrics['total_requests']} requests per variant")
        
        print(f"\n⚡ LATENCY METRICS:")
        print(f"  Challenger (A): {challenger_metrics['mean_latency_ms']:.1f}ms ± {challenger_metrics['latency_std']:.1f}ms")
        print(f"  Incumbent  (B): {incumbent_metrics['mean_latency_ms']:.1f}ms ± {incumbent_metrics['latency_std']:.1f}ms")
        print(f"  Improvement:    {latency_improvement:+.1f}% {'✅' if latency_improvement > 0 else '❌'}")
        print(f"  P95 Latency A:  {challenger_metrics['p95_latency_ms']:.1f}ms")
        print(f"  P95 Latency B:  {incumbent_metrics['p95_latency_ms']:.1f}ms")
        print(f"  Statistical Sig: {'✅ YES' if latency_sig else '❌ NO'} (p={latency_p_value:.4f})")
        
        print(f"\n🎯 ACCURACY METRICS:")
        print(f"  Challenger (A): {challenger_metrics['accuracy_rate']:.1%}")
        print(f"  Incumbent  (B): {incumbent_metrics['accuracy_rate']:.1%}")
        print(f"  Improvement:    {accuracy_improvement:+.1f}pp {'✅' if accuracy_improvement > 0 else '❌'}")
        print(f"  Statistical Sig: {'✅ YES' if accuracy_sig else '❌ NO'} (p={accuracy_p_value:.4f})")
        
        # Deployment recommendation
        print(f"\n🚀 DEPLOYMENT RECOMMENDATION:")
        
        latency_better = latency_improvement > 0 and latency_sig
        accuracy_better = accuracy_improvement > 0 and accuracy_sig
        no_regression = (latency_improvement > -5 or not latency_sig) and (accuracy_improvement > -2 or not accuracy_sig)
        
        if (latency_better or accuracy_better) and no_regression:
            print("  ✅ DEPLOY CHALLENGER")
            print("  Challenger shows statistically significant improvements")
            if latency_better and accuracy_better:
                print("  with better latency AND accuracy!")
            elif latency_better:
                print("  with significantly better latency!")
            elif accuracy_better:
                print("  with significantly better accuracy!")
        elif no_regression:
            print("  ⚖️  NEUTRAL - No significant differences detected")
            print("  Consider business factors for deployment decision")
        else:
            print("  ❌ KEEP INCUMBENT")
            print("  Challenger shows concerning regressions")
        
        print("="*70)


def main():
    """Run A/B test demonstration."""
    # Create challenger and incumbent agents with different characteristics
    challenger = MockSupportAgent(
        name="Challenger-GPT4o-Optimized", 
        base_latency=150,  # Faster
        accuracy_rate=0.92  # More accurate
    )
    
    incumbent = MockSupportAgent(
        name="Incumbent-GPT35-Turbo",
        base_latency=200,  # Slower
        accuracy_rate=0.88  # Less accurate
    )
    
    # Run A/B test
    ab_test = ABTestFramework(challenger, incumbent)
    results = ab_test.run_test(sample_size=50)
    
    # Print scorecard
    ab_test.print_scorecard(results)


if __name__ == "__main__":
    main()