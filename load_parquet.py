import pandas as pd
import glob
import sys
import io
import json
import os
import base64

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Find all parquet files in the 'parquet' directory
file_paths = glob.glob('parquet/*.parquet')

# Read all files into a list of dataframes
df_list = [pd.read_parquet(file) for file in file_paths]

# Concatenate all dataframes into one
if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)
    print("=" * 80)
    print("THÔNG TIN TỔNG QUAN VỀ DỮ LIỆU")
    print("=" * 80)
    print(f"\nTổng số dòng: {len(combined_df):,}")
    print(f"Tổng số cột: {len(combined_df.columns)}")
    print(f"\nTên các cột: {list(combined_df.columns)}")
    print("\n" + "-" * 80)
    print("KIỂU DỮ LIỆU CỦA CÁC CỘT:")
    print("-" * 80)
    print(combined_df.dtypes)
    print("\n" + "-" * 80)
    print("10 DÒNG ĐẦU TIÊN:")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(combined_df.head(10))
    print("\n" + "-" * 80)
    print("THỐNG KÊ MÔ TẢ:")
    print("-" * 80)
    print(combined_df.describe())
    
    # Export to JSON
    print("\n" + "=" * 80)
    print("ĐANG XUẤT DỮ LIỆU RA FILE JSON...")
    print("=" * 80)
    
    # Create a copy for JSON export
    df_for_json = combined_df.copy()
    
    # Handle image column - convert bytes to base64 string for JSON
    def convert_image_to_base64(image_data):
        if pd.isna(image_data) or image_data is None:
            return None
        if isinstance(image_data, dict) and 'bytes' in image_data:
            # Convert bytes to base64 string
            image_bytes = image_data['bytes']
            if isinstance(image_bytes, bytes):
                return base64.b64encode(image_bytes).decode('utf-8')
        return None
    
    if 'image' in df_for_json.columns:
        print("Đang chuyển đổi dữ liệu ảnh sang base64...")
        df_for_json['image'] = df_for_json['image'].apply(convert_image_to_base64)
    
    # Convert to JSON
    # Use orient='records' to get list of dictionaries
    json_data = df_for_json.to_dict(orient='records')
    
    # Save to JSON file
    output_file = 'parquet_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    # Get file size
    file_size = os.path.getsize(output_file) / 1024 / 1024  # Convert to MB
    
    print(f"\n✓ Đã xuất dữ liệu thành công vào file: {output_file}")
    print(f"✓ Tổng số bản ghi: {len(json_data):,}")
    print(f"✓ Kích thước file: {file_size:.2f} MB")
    
else:
    print("No .parquet files found in the 'parquet' directory.")

