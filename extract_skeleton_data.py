import os
import pandas as pd
import logging
from tqdm import tqdm  # For progress bar

logging.basicConfig(level=logging.INFO)

# Constants
PARQUET_DIR = '/home/ubuntu/v2/yoga/3DYoga90/data_process/precomputed_skeleton/official_dataset'
OUTPUT_SKELETAL_CSV_PATH = '/home/ubuntu/v2/yoga/csv_files/extracted_skeletal_data_with_sequence.csv'

def main():
    # Get list of all parquet files
    print("\nGetting list of parquet files...")
    parquet_files = [f for f in os.listdir(PARQUET_DIR) if f.endswith('.parquet')]
    parquet_files.sort()
    total_files = len(parquet_files)
    print(f"Found {total_files} parquet files to process")
    
    # Process first file to create CSV with headers
    first_file = parquet_files[0]
    sequence_id = int(first_file.replace('.parquet', ''))
    print(f"\nProcessing first file {first_file} with sequence_id {sequence_id}")
    
    # Read and process first file
    file_path = os.path.join(PARQUET_DIR, first_file)
    df = pd.read_parquet(file_path)
    df['sequence_id'] = sequence_id
    
    # Write first file with headers
    df.to_csv(OUTPUT_SKELETAL_CSV_PATH, index=False)
    print(f"Created CSV file with headers")
    
    # Process remaining files with progress bar
    print("\nProcessing remaining files:")
    for file in tqdm(parquet_files[1:], desc="Processing"):
        try:
            sequence_id = int(file.replace('.parquet', ''))
            
            # Read parquet file
            file_path = os.path.join(PARQUET_DIR, file)
            df = pd.read_parquet(file_path)
            
            # Add sequence_id
            df['sequence_id'] = sequence_id
            
            # Append to CSV without headers
            df.to_csv(OUTPUT_SKELETAL_CSV_PATH, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"\nError processing {file}: {e}")
    
    # Final verification
    print("\nProcessing complete. Verifying final output...")
    try:
        verify_df = pd.read_csv(OUTPUT_SKELETAL_CSV_PATH, nrows=5)
        print("\nFirst 5 rows of output file:")
        print(verify_df)
        
        # Count total rows
        with open(OUTPUT_SKELETAL_CSV_PATH, 'r') as f:
            total_rows = sum(1 for _ in f) - 1  # Subtract 1 for header
        
        print(f"\nTotal rows in output file: {total_rows}")
        print(f"Total files processed: {total_files}")
        print(f"\nOutput saved to: {OUTPUT_SKELETAL_CSV_PATH}")
        
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    # Remove output file if it exists
    if os.path.exists(OUTPUT_SKELETAL_CSV_PATH):
        os.remove(OUTPUT_SKELETAL_CSV_PATH)
        print("Removed existing output file")
    
    main()