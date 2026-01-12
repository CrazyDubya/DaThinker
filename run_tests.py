#!/usr/bin/env python3
"""
DaThinker Adaptive System Test Runner

Runs all five test scenarios and produces a comprehensive report.

Usage:
    python run_tests.py              # Run all scenarios
    python run_tests.py --scenario 1 # Run specific scenario
    python run_tests.py --quick      # Run quick tests only
    python run_tests.py --verbose    # Verbose output
"""

import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, '/home/user/DaThinker')

from tests.scenarios.scenario_1_manipulation import run_scenario as run_manipulation
from tests.scenarios.scenario_2_edge_cases import run_scenario as run_edge_cases
from tests.scenarios.scenario_3_perfect_conversation import run_scenario as run_perfect
from tests.scenarios.scenario_4_injections import run_scenario as run_injections
from tests.scenarios.scenario_5_illogical import run_scenario as run_illogical


class TestRunner:
    """Orchestrates running all test scenarios."""

    SCENARIOS = {
        1: ("Manipulation Attempts", run_manipulation),
        2: ("Edge Cases", run_edge_cases),
        3: ("Perfect Conversation", run_perfect),
        4: ("Injection Attacks", run_injections),
        5: ("Illogical Inputs", run_illogical),
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: Dict[int, Dict[str, Any]] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def run_all(self) -> Dict[str, Any]:
        """Run all scenarios and compile results."""
        self.start_time = datetime.now()

        print("\n" + "=" * 70)
        print("  DATHINKER ADAPTIVE SYSTEM TEST SUITE")
        print("=" * 70)
        print(f"  Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        for scenario_num, (name, run_func) in self.SCENARIOS.items():
            print(f"\n{'#' * 70}")
            print(f"# SCENARIO {scenario_num}: {name.upper()}")
            print(f"{'#' * 70}")

            try:
                result = run_func()
                self.results[scenario_num] = {
                    "name": name,
                    "status": "completed",
                    "results": result
                }
            except Exception as e:
                self.results[scenario_num] = {
                    "name": name,
                    "status": "error",
                    "error": str(e)
                }
                print(f"\n  ERROR in scenario {scenario_num}: {e}")

        self.end_time = datetime.now()
        return self._compile_report()

    def run_scenario(self, scenario_num: int) -> Dict[str, Any]:
        """Run a specific scenario."""
        if scenario_num not in self.SCENARIOS:
            raise ValueError(f"Invalid scenario number: {scenario_num}. Valid: 1-5")

        name, run_func = self.SCENARIOS[scenario_num]
        self.start_time = datetime.now()

        print(f"\n{'#' * 70}")
        print(f"# SCENARIO {scenario_num}: {name.upper()}")
        print(f"{'#' * 70}")

        try:
            result = run_func()
            self.results[scenario_num] = {
                "name": name,
                "status": "completed",
                "results": result
            }
        except Exception as e:
            self.results[scenario_num] = {
                "name": name,
                "status": "error",
                "error": str(e)
            }

        self.end_time = datetime.now()
        return self._compile_report()

    def _compile_report(self) -> Dict[str, Any]:
        """Compile a comprehensive test report."""
        duration = (self.end_time - self.start_time).total_seconds()

        report = {
            "test_suite": "DaThinker Adaptive System",
            "run_time": {
                "started": self.start_time.isoformat(),
                "ended": self.end_time.isoformat(),
                "duration_seconds": round(duration, 2)
            },
            "scenarios": self.results,
            "summary": self._calculate_summary()
        }

        self._print_report(report)
        return report

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate overall summary statistics."""
        completed = sum(1 for r in self.results.values() if r["status"] == "completed")
        errors = sum(1 for r in self.results.values() if r["status"] == "error")

        total_tests = 0
        total_passed = 0

        for result in self.results.values():
            if result["status"] == "completed" and "results" in result:
                res = result["results"]

                # Handle different result structures
                if "individual" in res:
                    ind = res["individual"]
                    if "total_tests" in ind:
                        total_tests += ind["total_tests"]
                    if "passed" in ind:
                        total_passed += ind["passed"]
                    elif "handled" in ind:
                        total_passed += ind["handled"]
                    elif "blocked_correctly" in ind:
                        total_passed += ind["blocked_correctly"]

                if "flows" in res:
                    flows = res["flows"]
                    if "total_flows" in flows:
                        total_tests += flows["total_flows"]
                    if "passed" in flows:
                        total_passed += flows["passed"]

        return {
            "scenarios_completed": completed,
            "scenarios_errored": errors,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "overall_pass_rate": round(total_passed / total_tests * 100, 1) if total_tests > 0 else 0
        }

    def _print_report(self, report: Dict[str, Any]):
        """Print a formatted test report."""
        print("\n" + "=" * 70)
        print("  FINAL TEST REPORT")
        print("=" * 70)

        summary = report["summary"]
        print(f"\n  Scenarios Completed: {summary['scenarios_completed']}/{len(self.SCENARIOS)}")
        print(f"  Scenarios Errored: {summary['scenarios_errored']}")
        print(f"  Total Tests Run: {summary['total_tests']}")
        print(f"  Total Tests Passed: {summary['total_passed']}")
        print(f"  Overall Pass Rate: {summary['overall_pass_rate']}%")
        print(f"  Duration: {report['run_time']['duration_seconds']}s")

        print("\n  Scenario Breakdown:")
        print("  " + "-" * 66)

        for scenario_num, result in self.results.items():
            status_icon = "✓" if result["status"] == "completed" else "✗"
            print(f"  {status_icon} Scenario {scenario_num}: {result['name']}")

            if result["status"] == "error":
                print(f"      Error: {result.get('error', 'Unknown')}")
            elif "results" in result:
                res = result["results"]
                if "individual" in res:
                    ind = res["individual"]
                    if "success_rate" in ind:
                        print(f"      Success Rate: {ind['success_rate']:.1f}%")
                    elif "block_rate" in ind:
                        print(f"      Block Rate: {ind['block_rate']:.1f}%")

        print("\n" + "=" * 70)
        print("  TEST RUN COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run DaThinker Adaptive System Tests"
    )
    parser.add_argument(
        "--scenario", "-s",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run a specific scenario (1-5)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to file"
    )

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose)

    if args.scenario:
        report = runner.run_scenario(args.scenario)
    else:
        report = runner.run_all()

    if args.json:
        print("\n" + json.dumps(report, indent=2, default=str))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Results saved to: {args.output}")

    # Exit with appropriate code
    if report["summary"]["scenarios_errored"] > 0:
        sys.exit(1)
    elif report["summary"]["overall_pass_rate"] < 70:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
