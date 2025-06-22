#!/usr/bin/env python3
"""
Hugging Face Dataset Deployment Verification Script
=================================================

Comprehensive verification suite for the AHGD dataset deployment.
Tests all aspects of the deployed dataset including accessibility,
format compatibility, and functionality.

Usage:
    python scripts/verify_deployment.py
    python scripts/verify_deployment.py --repo-id massomo/ahgd
    python scripts/verify_deployment.py --detailed
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime, timezone

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import requests
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, dataset_info, whoami
from huggingface_hub.utils import RepositoryNotFoundError

from src.utils.logging import get_logger
from src.monitoring.analytics import DatasetAnalytics, create_monitoring_system

logger = get_logger(__name__)


class DeploymentVerifier:
    """Comprehensive verification suite for AHGD dataset deployment."""
    
    def __init__(self, repo_id: str = "massomo/ahgd"):
        self.repo_id = repo_id
        self.api = HfApi()
        self.verification_results = {
            "repo_id": repo_id,
            "verification_timestamp": datetime.now(timezone.utc).isoformat(),
            "tests": {},
            "overall_status": "unknown",
            "detailed_results": {}
        }
    
    def verify_authentication(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify Hugging Face authentication."""
        try:
            user_info = whoami()
            result = {
                "authenticated": True,
                "user": user_info.get('name', 'unknown'),
                "user_type": user_info.get('type', 'user')
            }
            logger.info(f"Authentication verified: {result['user']}")
            return True, result
        except Exception as e:
            result = {
                "authenticated": False,
                "error": str(e),
                "suggestion": "Run: huggingface-cli login"
            }
            logger.warning(f"Authentication failed: {e}")
            return False, result
    
    def verify_repository_exists(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify that the repository exists and is accessible."""
        try:
            info = dataset_info(self.repo_id)
            result = {
                "exists": True,
                "repo_id": self.repo_id,
                "created_at": info.created_at.isoformat() if info.created_at else None,
                "last_modified": info.last_modified.isoformat() if info.last_modified else None,
                "downloads": getattr(info, 'downloads', 0),
                "likes": getattr(info, 'likes', 0),
                "tags": getattr(info, 'tags', []),
                "private": getattr(info, 'private', False)
            }
            logger.info(f"Repository verified: {self.repo_id}")
            return True, result
        except RepositoryNotFoundError:
            result = {
                "exists": False,
                "error": f"Repository {self.repo_id} not found",
                "suggestion": "Check repository name or create the repository"
            }
            logger.error(f"Repository not found: {self.repo_id}")
            return False, result
        except Exception as e:
            result = {
                "exists": False,
                "error": str(e),
                "suggestion": "Check repository permissions and connectivity"
            }
            logger.error(f"Repository verification failed: {e}")
            return False, result
    
    def verify_dataset_loading(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify that the dataset can be loaded using the datasets library."""
        try:
            logger.info("Loading dataset using datasets library...")
            dataset = load_dataset(self.repo_id)
            
            # Analyse dataset structure
            result = {
                "loadable": True,
                "splits": list(dataset.keys()),
                "total_rows": sum(len(split) for split in dataset.values()),
                "features": {},
                "sample_data": {}
            }
            
            # Get features and sample data for each split
            for split_name, split_data in dataset.items():
                result["features"][split_name] = list(split_data.features.keys())
                if len(split_data) > 0:
                    # Get first row as sample
                    sample = split_data[0]
                    result["sample_data"][split_name] = {
                        k: str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                        for k, v in sample.items()
                    }
            
            logger.info(f"Dataset loaded successfully: {result['total_rows']} rows")
            return True, result
            
        except Exception as e:
            result = {
                "loadable": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "suggestion": "Check dataset format and repository configuration"
            }
            logger.error(f"Dataset loading failed: {e}")
            return False, result
    
    def verify_file_formats(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify all expected file formats are present and accessible."""
        expected_files = [
            "README.md",
            "ahgd_data.parquet",
            "ahgd_data.csv", 
            "ahgd_data.json",
            "ahgd_data.geojson",
            "data_dictionary.json",
            "dataset_metadata.json"
        ]
        
        result = {
            "files_verified": {},
            "all_files_present": True,
            "missing_files": [],
            "accessible_files": []
        }
        
        try:
            # Get repository file listing
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            
            for expected_file in expected_files:
                if expected_file in files:
                    result["files_verified"][expected_file] = {
                        "present": True,
                        "accessible": True
                    }
                    result["accessible_files"].append(expected_file)
                else:
                    result["files_verified"][expected_file] = {
                        "present": False,
                        "accessible": False
                    }
                    result["missing_files"].append(expected_file)
                    result["all_files_present"] = False
            
            # Check for additional files
            additional_files = set(files) - set(expected_files)
            if additional_files:
                result["additional_files"] = list(additional_files)
            
            logger.info(f"File verification completed. Missing: {len(result['missing_files'])}")
            return result["all_files_present"], result
            
        except Exception as e:
            result.update({
                "error": str(e),
                "suggestion": "Check repository permissions and file upload status"
            })
            logger.error(f"File format verification failed: {e}")
            return False, result
    
    def verify_metadata_quality(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify dataset metadata quality and completeness."""
        result = {
            "readme_present": False,
            "dataset_card_complete": False,
            "metadata_valid": False,
            "license_specified": False,
            "tags_present": False
        }
        
        try:
            # Check README/dataset card
            try:
                readme_content = self.api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename="README.md",
                    repo_type="dataset"
                )
                with open(readme_content, 'r', encoding='utf-8') as f:
                    readme_text = f.read()
                
                result["readme_present"] = True
                result["readme_length"] = len(readme_text)
                
                # Check for essential sections
                essential_sections = ["Dataset Description", "Usage", "License"]
                result["essential_sections"] = {}
                for section in essential_sections:
                    result["essential_sections"][section] = section.lower() in readme_text.lower()
                
                result["dataset_card_complete"] = all(result["essential_sections"].values())
                
            except Exception as e:
                result["readme_error"] = str(e)
            
            # Check dataset metadata
            try:
                metadata_content = self.api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename="dataset_metadata.json",
                    repo_type="dataset"
                )
                with open(metadata_content, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                result["metadata_valid"] = True
                result["metadata_structure"] = list(metadata.keys())
                
                # Check license information
                if "dataset_info" in metadata and "license" in metadata["dataset_info"]:
                    result["license_specified"] = True
                    result["license"] = metadata["dataset_info"]["license"]
                
            except Exception as e:
                result["metadata_error"] = str(e)
            
            # Check repository tags
            try:
                info = dataset_info(self.repo_id)
                tags = getattr(info, 'tags', [])
                result["tags_present"] = len(tags) > 0
                result["tags"] = tags
            except Exception as e:
                result["tags_error"] = str(e)
            
            # Overall metadata quality score
            quality_checks = [
                result["readme_present"],
                result["dataset_card_complete"],
                result["metadata_valid"],
                result["license_specified"],
                result["tags_present"]
            ]
            result["metadata_quality_score"] = sum(quality_checks) / len(quality_checks)
            
            success = result["metadata_quality_score"] >= 0.8
            logger.info(f"Metadata quality score: {result['metadata_quality_score']:.2f}")
            return success, result
            
        except Exception as e:
            result.update({
                "error": str(e),
                "suggestion": "Check metadata files and repository configuration"
            })
            logger.error(f"Metadata verification failed: {e}")
            return False, result
    
    def verify_data_formats(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify that all data formats can be loaded and are consistent."""
        result = {
            "formats_tested": {},
            "all_formats_valid": True,
            "consistency_check": {}
        }
        
        formats_to_test = {
            "parquet": "ahgd_data.parquet",
            "csv": "ahgd_data.csv",
            "json": "ahgd_data.json"
        }
        
        data_frames = {}
        
        for format_name, filename in formats_to_test.items():
            try:
                logger.info(f"Testing {format_name} format...")
                
                # Download file
                file_path = self.api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    repo_type="dataset"
                )
                
                # Load data based on format
                if format_name == "parquet":
                    df = pd.read_parquet(file_path)
                elif format_name == "csv":
                    df = pd.read_csv(file_path)
                elif format_name == "json":
                    df = pd.read_json(file_path)
                
                data_frames[format_name] = df
                
                result["formats_tested"][format_name] = {
                    "loadable": True,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                    "file_size_mb": os.path.getsize(file_path) / (1024 * 1024)
                }
                
            except Exception as e:
                result["formats_tested"][format_name] = {
                    "loadable": False,
                    "error": str(e)
                }
                result["all_formats_valid"] = False
        
        # Check consistency between formats
        if len(data_frames) > 1:
            reference_df = list(data_frames.values())[0]
            reference_format = list(data_frames.keys())[0]
            
            for format_name, df in data_frames.items():
                if format_name != reference_format:
                    consistency = {
                        "same_shape": df.shape == reference_df.shape,
                        "same_columns": list(df.columns) == list(reference_df.columns)
                    }
                    
                    # Check data consistency (sample comparison)
                    if consistency["same_shape"] and consistency["same_columns"]:
                        try:
                            # Compare first few rows
                            sample_consistent = df.head().equals(reference_df.head())
                            consistency["sample_data_consistent"] = sample_consistent
                        except Exception:
                            consistency["sample_data_consistent"] = False
                    
                    result["consistency_check"][f"{reference_format}_vs_{format_name}"] = consistency
        
        logger.info(f"Data format verification completed. Valid formats: {sum(1 for f in result['formats_tested'].values() if f.get('loadable', False))}")
        return result["all_formats_valid"], result
    
    def verify_examples_functionality(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify that example code works correctly."""
        result = {
            "examples_tested": {},
            "all_examples_work": True
        }
        
        # Test basic Python example
        try:
            logger.info("Testing basic Python usage example...")
            
            # This simulates the basic usage pattern
            dataset = load_dataset(self.repo_id)
            df = dataset['train'].to_pandas()
            
            result["examples_tested"]["basic_python"] = {
                "works": True,
                "dataset_loaded": True,
                "pandas_conversion": True,
                "rows_loaded": len(df)
            }
            
        except Exception as e:
            result["examples_tested"]["basic_python"] = {
                "works": False,
                "error": str(e)
            }
            result["all_examples_work"] = False
        
        # Test format-specific loading
        for format_name in ["csv", "json", "parquet"]:
            try:
                logger.info(f"Testing {format_name} format loading...")
                
                # Download and test format
                file_path = self.api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename=f"ahgd_data.{format_name}",
                    repo_type="dataset"
                )
                
                if format_name == "csv":
                    test_df = pd.read_csv(file_path)
                elif format_name == "json":
                    test_df = pd.read_json(file_path)
                elif format_name == "parquet":
                    test_df = pd.read_parquet(file_path)
                
                result["examples_tested"][f"{format_name}_loading"] = {
                    "works": True,
                    "rows": len(test_df),
                    "columns": len(test_df.columns)
                }
                
            except Exception as e:
                result["examples_tested"][f"{format_name}_loading"] = {
                    "works": False,
                    "error": str(e)
                }
                result["all_examples_work"] = False
        
        logger.info(f"Example functionality verification completed. Working examples: {sum(1 for e in result['examples_tested'].values() if e.get('works', False))}")
        return result["all_examples_work"], result
    
    def run_comprehensive_verification(self, detailed: bool = False) -> Dict[str, Any]:
        """Run all verification tests and compile results."""
        logger.info("Starting comprehensive deployment verification...")
        
        # Run all verification tests
        tests = [
            ("authentication", self.verify_authentication),
            ("repository_exists", self.verify_repository_exists),
            ("dataset_loading", self.verify_dataset_loading),
            ("file_formats", self.verify_file_formats),
            ("metadata_quality", self.verify_metadata_quality),
            ("data_formats", self.verify_data_formats),
            ("examples_functionality", self.verify_examples_functionality)
        ]
        
        for test_name, test_function in tests:
            try:
                logger.info(f"Running {test_name} verification...")
                success, details = test_function()
                
                self.verification_results["tests"][test_name] = {
                    "success": success,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                if detailed:
                    self.verification_results["detailed_results"][test_name] = details
                else:
                    # Include summary information
                    if "error" in details:
                        self.verification_results["tests"][test_name]["error"] = details["error"]
                    if "suggestion" in details:
                        self.verification_results["tests"][test_name]["suggestion"] = details["suggestion"]
                
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                self.verification_results["tests"][test_name] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
        
        # Calculate overall status
        successful_tests = sum(1 for test in self.verification_results["tests"].values() if test["success"])
        total_tests = len(self.verification_results["tests"])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        if success_rate >= 0.9:
            self.verification_results["overall_status"] = "success"
        elif success_rate >= 0.7:
            self.verification_results["overall_status"] = "warning"
        else:
            self.verification_results["overall_status"] = "failure"
        
        self.verification_results["summary"] = {
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "critical_failures": [
                name for name, result in self.verification_results["tests"].items()
                if not result["success"] and name in ["repository_exists", "dataset_loading"]
            ]
        }
        
        logger.info(f"Verification completed. Status: {self.verification_results['overall_status']} ({successful_tests}/{total_tests} tests passed)")
        
        return self.verification_results
    
    def save_verification_report(self, output_path: str = "data_exports/verification_report.json"):
        """Save verification results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.verification_results, f, indent=2)
        
        logger.info(f"Verification report saved to: {output_file}")


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(description="Verify AHGD dataset deployment on Hugging Face Hub")
    parser.add_argument("--repo-id", default="massomo/ahgd", help="Repository ID to verify")
    parser.add_argument("--detailed", action="store_true", help="Include detailed test results")
    parser.add_argument("--output", default="data_exports/verification_report.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Run verification
    verifier = DeploymentVerifier(repo_id=args.repo_id)
    results = verifier.run_comprehensive_verification(detailed=args.detailed)
    
    # Save results
    verifier.save_verification_report(output_path=args.output)
    
    # Print summary
    print(f"\n=== AHGD Dataset Deployment Verification ===")
    print(f"Repository: {args.repo_id}")
    print(f"Overall Status: {results['overall_status'].upper()}")
    print(f"Tests Passed: {results['summary']['successful_tests']}/{results['summary']['total_tests']}")
    print(f"Success Rate: {results['summary']['success_rate']:.1%}")
    
    if results['summary']['critical_failures']:
        print(f"Critical Failures: {', '.join(results['summary']['critical_failures'])}")
    
    print(f"\nDetailed results saved to: {args.output}")
    
    # Exit with appropriate code
    if results['overall_status'] == 'success':
        return 0
    elif results['overall_status'] == 'warning':
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit(main())