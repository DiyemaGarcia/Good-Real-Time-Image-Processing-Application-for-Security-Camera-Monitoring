"""
Download pretrained models for SuperPoint and D2-Net
These are the exact models referenced in feature detection research
"""
import urllib.request
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, destination: str) -> bool:
    """
    Download file from URL with progress indication
    
    Args:
        url: URL to download from
        destination: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        file_size = dest_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"File already exists: {destination} ({file_size:.2f} MB)")
        return True
    
    logger.info(f"Downloading from {url}")
    logger.info(f"Destination: {destination}")
    
    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            sys.stdout.write(f"\r  Progress: {percent:.1f}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print()  # New line after progress
        
        file_size = dest_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✓ Successfully downloaded {destination} ({file_size:.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download {url}: {str(e)}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial download
        return False


def download_pretrained_models():
    """
    Download all pretrained models required for the experiment
    
    Models:
    1. SuperPoint (MagicLeap) - Self-supervised interest point detector
    2. D2-Net (CVPR 2019) - Joint detection and description network
    """
    models_dir = Path("pretrained_models")
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("DOWNLOADING PRETRAINED MODELS FOR FEATURE DETECTION")
    print("=" * 80)
    print()
    
    models = [
        {
            'name': 'SuperPoint',
            'url': 'https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth',
            'path': models_dir / 'superpoint_v1.pth',
            'description': 'SuperPoint pretrained on MS-COCO (MagicLeap)'
        },
        {
            'name': 'D2-Net',
            'url': 'https://dusmanu.com/files/d2-net/d2_tf.pth',
            'path': models_dir / 'd2_tf.pth',
            'description': 'D2-Net pretrained on MegaDepth (CVPR 2019)'
        }
    ]
    
    results = []
    for model in models:
        print(f"\n{model['name']}:")
        print(f"  Description: {model['description']}")
        print("-" * 80)
        
        success = download_file(model['url'], str(model['path']))
        results.append((model['name'], success))
    
    print()
    print("=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    
    all_success = True
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name}: {status}")
        if not success:
            all_success = False
    
    print()
    if all_success:
        print("✓ All models downloaded successfully!")
        print("  You can now run the feature detection experiments.")
    else:
        print("⚠ Some models failed to download.")
        print("  Please check your internet connection and try again.")
    
    print("=" * 80)
    
    return all_success


if __name__ == "__main__":
    download_pretrained_models()