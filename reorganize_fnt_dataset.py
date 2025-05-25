#!/usr/bin/env python3
"""
Script to reorganize Chars74K English/Fnt dataset for alphabetic recognition.
Converts Sample folders to case-safe character folders for Windows compatibility.

Dataset structure:
- Sample001-010 → 0_digit, 1_digit, ..., 9_digit
- Sample011-036 → A_upper, B_upper, ..., Z_upper  
- Sample037-062 → a_lower, b_lower, ..., z_lower
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Dataset paths
SOURCE_DIR = Path("d:/Development/Proyek/Citra/Project_UAS/dataset/alphabets")
TARGET_DIR = Path("d:/Development/Proyek/Citra/Project_UAS/dataset/fnt_organized")

# Mapping from Sample folders to character labels (case-safe)
FNT_CLASS_MAPPING = {
    # Digits (Sample001-010)
    'Sample001': '0_digit',
    'Sample002': '1_digit',
    'Sample003': '2_digit',
    'Sample004': '3_digit',
    'Sample005': '4_digit',
    'Sample006': '5_digit',
    'Sample007': '6_digit',
    'Sample008': '7_digit',
    'Sample009': '8_digit',
    'Sample010': '9_digit',
    
    # Uppercase letters (Sample011-036)
    'Sample011': 'A_upper',
    'Sample012': 'B_upper',
    'Sample013': 'C_upper',
    'Sample014': 'D_upper',
    'Sample015': 'E_upper',
    'Sample016': 'F_upper',
    'Sample017': 'G_upper',
    'Sample018': 'H_upper',
    'Sample019': 'I_upper',
    'Sample020': 'J_upper',
    'Sample021': 'K_upper',
    'Sample022': 'L_upper',
    'Sample023': 'M_upper',
    'Sample024': 'N_upper',
    'Sample025': 'O_upper',
    'Sample026': 'P_upper',
    'Sample027': 'Q_upper',
    'Sample028': 'R_upper',
    'Sample029': 'S_upper',
    'Sample030': 'T_upper',
    'Sample031': 'U_upper',
    'Sample032': 'V_upper',
    'Sample033': 'W_upper',
    'Sample034': 'X_upper',
    'Sample035': 'Y_upper',
    'Sample036': 'Z_upper',
    
    # Lowercase letters (Sample037-062)
    'Sample037': 'a_lower',
    'Sample038': 'b_lower',
    'Sample039': 'c_lower',
    'Sample040': 'd_lower',
    'Sample041': 'e_lower',
    'Sample042': 'f_lower',
    'Sample043': 'g_lower',
    'Sample044': 'h_lower',
    'Sample045': 'i_lower',
    'Sample046': 'j_lower',
    'Sample047': 'k_lower',
    'Sample048': 'l_lower',
    'Sample049': 'm_lower',
    'Sample050': 'n_lower',
    'Sample051': 'o_lower',
    'Sample052': 'p_lower',
    'Sample053': 'q_lower',
    'Sample054': 'r_lower',
    'Sample055': 's_lower',
    'Sample056': 't_lower',
    'Sample057': 'u_lower',
    'Sample058': 'v_lower',
    'Sample059': 'w_lower',
    'Sample060': 'x_lower',
    'Sample061': 'y_lower',
    'Sample062': 'z_lower'
}

def create_target_directories():
    """Create target directory structure."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each character class
    for char_folder in FNT_CLASS_MAPPING.values():
        char_dir = TARGET_DIR / char_folder
        char_dir.mkdir(exist_ok=True)
    
    print(f"Created target directory: {TARGET_DIR}")
    print(f"Created {len(FNT_CLASS_MAPPING)} character class folders")

def copy_images():
    """Copy images from Sample folders to character folders."""
    total_images = 0
    
    for sample_folder, char_folder in tqdm(FNT_CLASS_MAPPING.items(), desc="Processing Sample folders"):
        source_path = SOURCE_DIR / sample_folder
        target_path = TARGET_DIR / char_folder
        
        if not source_path.exists():
            print(f"Warning: Source folder {source_path} does not exist")
            continue
        
        # Get all PNG files from source folder
        png_files = list(source_path.glob("*.png"))
        
        if not png_files:
            print(f"Warning: No PNG files found in {source_path}")
            continue
        
        # Copy each PNG file
        for png_file in png_files:
            target_file = target_path / png_file.name
            shutil.copy2(png_file, target_file)
            total_images += 1
        
        print(f"Copied {len(png_files)} images from {sample_folder} to {char_folder}")
    
    return total_images

def verify_organization():
    """Verify the reorganized dataset."""
    print("\n=== Dataset Verification ===")
    
    total_images = 0
    for char_folder in sorted(FNT_CLASS_MAPPING.values()):
        char_dir = TARGET_DIR / char_folder
        
        if char_dir.exists():
            png_files = list(char_dir.glob("*.png"))
            total_images += len(png_files)
            print(f"{char_folder}: {len(png_files)} images")
        else:
            print(f"{char_folder}: Directory not found!")
    
    print(f"\nTotal images: {total_images}")
    
    # Check if we have expected number of images (62,992 total, ~1,016 per class)
    expected_total = 62992
    if total_images == expected_total:
        print("✅ Dataset organization successful!")
    else:
        print(f"⚠️ Expected {expected_total} images, found {total_images}")

def main():
    """Main reorganization process."""
    print("=== Chars74K English/Fnt Dataset Reorganization ===")
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print(f"Classes: {len(FNT_CLASS_MAPPING)} (0-9, A-Z, a-z)")
    
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} does not exist!")
        return
    
    # Ask for confirmation
    response = input("\nProceed with reorganization? (y/N): ").strip().lower()
    if response != 'y':
        print("Operation cancelled.")
        return
    
    try:
        # Step 1: Create directory structure
        print("\n1. Creating target directories...")
        create_target_directories()
        
        # Step 2: Copy images
        print("\n2. Copying images...")
        total_copied = copy_images()
        print(f"Total images copied: {total_copied}")
        
        # Step 3: Verify organization
        print("\n3. Verifying organization...")
        verify_organization()
        
        print("\n=== Reorganization Complete ===")
        print(f"Dataset ready for training at: {TARGET_DIR}")
        
    except Exception as e:
        print(f"Error during reorganization: {e}")
        return

if __name__ == "__main__":
    main()
