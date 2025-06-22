#!/usr/bin/env python3
"""
Quick deployment script for AHGD dataset to Hugging Face Hub
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, whoami, login

def main():
    print("🚀 AHGD Dataset Deployment to Hugging Face Hub")
    print("=" * 50)
    
    # Check authentication
    try:
        user = whoami()
        print(f"✅ Authenticated as: {user['name']}")
    except Exception as e:
        print("❌ Not authenticated with Hugging Face Hub")
        print("\nTo authenticate, you need to:")
        print("1. Get your access token from: https://huggingface.co/settings/tokens")
        print("2. Run: huggingface-cli login")
        print("   OR")
        print("3. Set environment variable: export HUGGING_FACE_HUB_TOKEN=your_token_here")
        
        # Try to use token from environment
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            print("\n🔑 Found token in environment, attempting login...")
            try:
                login(token=token)
                print("✅ Login successful!")
            except Exception as e:
                print(f"❌ Login failed: {e}")
                return 1
        else:
            print("\n⚠️  Please authenticate first and then run this script again.")
            return 1
    
    # Repository details
    repo_id = "massomo/ahgd"
    dataset_path = Path("data_exports/huggingface_dataset")
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_path}")
        return 1
    
    print(f"\n📁 Dataset location: {dataset_path}")
    print(f"🎯 Target repository: {repo_id}")
    
    # Initialize API
    api = HfApi()
    
    # Create or verify repository
    try:
        print(f"\n🔍 Checking if repository exists...")
        api.dataset_info(repo_id)
        print(f"✅ Repository {repo_id} already exists")
    except Exception:
        print(f"📝 Creating new repository: {repo_id}")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                exist_ok=True
            )
            print(f"✅ Repository created successfully")
        except Exception as e:
            print(f"❌ Failed to create repository: {e}")
            return 1
    
    # Upload dataset files
    print(f"\n📤 Uploading dataset files...")
    try:
        # List files to upload
        files = list(dataset_path.rglob("*"))
        files = [f for f in files if f.is_file()]
        print(f"📊 Found {len(files)} files to upload")
        
        # Upload folder
        print("🔄 Starting upload...")
        api.upload_folder(
            folder_path=str(dataset_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="🎉 Initial deployment of Australian Health and Geographic Data (AHGD) dataset"
        )
        
        print("✅ Upload completed successfully!")
        
        # Provide dataset URL
        dataset_url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"\n🎉 Dataset successfully deployed!")
        print(f"🌐 View your dataset at: {dataset_url}")
        
        # Test loading
        print(f"\n🧪 Testing dataset loading...")
        try:
            dataset = load_dataset(repo_id, split="train")
            print(f"✅ Dataset loads successfully! Found {len(dataset)} records")
        except Exception as e:
            print(f"⚠️  Dataset loading test failed: {e}")
            print("   This might be normal if the dataset is still processing.")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return 1
    
    print("\n✨ Deployment complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())