# NEW COMPLEMENT DICTIONARY
# Cấu trúc: Mỗi rule đảm bảo đủ 4 items (Tops + Bottoms + Shoes + Accessories)
# hoặc 3 items nếu không có Bottoms (như Dresses)

complement = {
    # ===== TOPS =====
    'Tshirts': [
        # Men combinations (4 items)
        ['Watches', 'Jeans', 'Casual Shoes'],
        ['Watches', 'Jeans', 'Sports Shoes'],
        ['Watches', 'Trousers', 'Casual Shoes'],
        ['Watches', 'Trousers', 'Formal Shoes'],
        ['Watches', 'Shorts', 'Sports Shoes'],
        ['Watches', 'Shorts', 'Casual Shoes'],
        # Women combinations (4 items)
        ['Watches', 'Skirts', 'Flats'],
        ['Watches', 'Skirts', 'Heels'],
        ['Watches', 'Jeans', 'Flats'],
        ['Handbags', 'Skirts', 'Casual Shoes'],
    ],
    
    'Shirts': [
        # Men formal (4 items)
        ['Watches', 'Trousers', 'Formal Shoes'],
        ['Belts', 'Trousers', 'Formal Shoes'],
        ['Watches', 'Jeans', 'Casual Shoes'],
        ['Belts', 'Jeans', 'Casual Shoes'],
        # Men casual (4 items)
        ['Watches', 'Shorts', 'Casual Shoes'],
        ['Watches', 'Trousers', 'Casual Shoes'],
    ],
    
    'Tops': [
        # Women combinations (4 items)
        ['Watches', 'Jeans', 'Casual Shoes'],
        ['Watches', 'Trousers', 'Casual Shoes'],
        ['Watches', 'Skirts', 'Flats'],
        ['Watches', 'Skirts', 'Heels'],
        ['Handbags', 'Shorts', 'Casual Shoes'],
        ['Watches', 'Capris', 'Sports Shoes'],
    ],
    
    'Sweaters': [
        ['Watches', 'Jeans', 'Casual Shoes'],
        ['Watches', 'Trousers', 'Formal Shoes'],
        ['Watches', 'Skirts', 'Flats'],  # Women
    ],
    
    'Sweatshirts': [
        ['Watches', 'Jeans', 'Sports Shoes'],
        ['Caps', 'Shorts', 'Sports Shoes'],
        ['Watches', 'Track Pants', 'Sports Shoes'],
        ['Backpacks', 'Trousers', 'Casual Shoes'],
    ],
    
    'Jackets': [
        ['Watches', 'Jeans', 'Casual Shoes'],
        ['Watches', 'Trousers', 'Formal Shoes'],
        ['Watches', 'Skirts', 'Heels'],  # Women
    ],
    
    # ===== DRESSES (Women only - 3 items vì không có Bottoms) =====
    'Dresses': [
        ['Watches', 'Heels'],
        ['Watches', 'Flats'],
        ['Handbags', 'Heels'],
        ['Handbags', 'Flats'],
        ['Watches', 'Casual Shoes'],
    ],
    
    # ===== BOTTOMS =====
    'Jeans': [
        ['Tshirts', 'Watches', 'Casual Shoes'],
        ['Shirts', 'Watches', 'Casual Shoes'],
        ['Tops', 'Watches', 'Casual Shoes'],  # Women
        ['Tshirts', 'Watches', 'Sports Shoes'],
        ['Sweaters', 'Watches', 'Casual Shoes'],
    ],
    
    'Trousers': [
        ['Shirts', 'Watches', 'Formal Shoes'],
        ['Shirts', 'Belts', 'Formal Shoes'],
        ['Tshirts', 'Watches', 'Casual Shoes'],
        ['Sweaters', 'Watches', 'Formal Shoes'],
        ['Tops', 'Watches', 'Casual Shoes'],  # Women
    ],
    
    'Shorts': [
        ['Tshirts', 'Watches', 'Sports Shoes'],
        ['Tshirts', 'Watches', 'Casual Shoes'],
        ['Tops', 'Watches', 'Sports Shoes'],  # Women
        ['Sweatshirts', 'Caps', 'Sports Shoes'],
    ],
    
    'Skirts': [
        # Women only (4 items)
        ['Tshirts', 'Watches', 'Flats'],
        ['Tshirts', 'Watches', 'Heels'],
        ['Tops', 'Watches', 'Flats'],
        ['Tops', 'Handbags', 'Heels'],
        ['Tshirts', 'Handbags', 'Casual Shoes'],
    ],
    
    'Capris': [
        # Women only (4 items)
        ['Tops', 'Watches', 'Sports Shoes'],
        ['Tshirts', 'Caps', 'Sports Shoes'],
    ],
    
    'Track Pants': [
        ['Tshirts', 'Watches', 'Sports Shoes'],
        ['Sweatshirts', 'Watches', 'Sports Shoes'],
        ['Tops', 'Watches', 'Sports Shoes'],  # Women
    ],
    
    # ===== SHOES =====
    'Casual Shoes': [
        ['Tshirts', 'Watches', 'Jeans'],
        ['Shirts', 'Watches', 'Trousers'],
        ['Tops', 'Watches', 'Skirts'],  # Women
    ],
    
    'Formal Shoes': [
        ['Shirts', 'Watches', 'Trousers'],
        ['Shirts', 'Belts', 'Trousers'],
    ],
    
    'Sports Shoes': [
        ['Tshirts', 'Watches', 'Shorts'],
        ['Tshirts', 'Watches', 'Track Pants'],
        ['Sweatshirts', 'Caps', 'Shorts'],
        ['Tops', 'Watches', 'Capris'],  # Women
    ],
    
    'Heels': [
        # Women only (3-4 items)
        ['Dresses', 'Watches'],
        ['Tshirts', 'Watches', 'Skirts'],
        ['Tops', 'Handbags', 'Skirts'],
    ],
    
    'Flats': [
        # Women only (3-4 items)
        ['Dresses', 'Watches'],
        ['Tshirts', 'Watches', 'Skirts'],
        ['Tops', 'Watches', 'Jeans'],
        ['Dresses', 'Handbags'],
    ],
    
    'Flip Flops': [
        ['Tshirts', 'Watches', 'Jeans'],
        ['Tshirts', 'Watches', 'Shorts'],
        ['Dresses', 'Handbags'],  # Women
    ],
    
    'Sandals': [
        ['Tshirts', 'Watches', 'Shorts'],
        ['Tshirts', 'Watches', 'Jeans'],
        ['Tops', 'Watches', 'Skirts'],  # Women
    ],
    
    # ===== ACCESSORIES =====
    'Watches': [
        ['Tshirts', 'Jeans', 'Casual Shoes'],
        ['Shirts', 'Trousers', 'Formal Shoes'],
        ['Tops', 'Skirts', 'Flats'],  # Women
        ['Dresses', 'Heels'],  # Women
    ],
    
    'Handbags': [
        # Women only (3-4 items)
        ['Dresses', 'Heels'],
        ['Dresses', 'Flats'],
        ['Tshirts', 'Skirts', 'Casual Shoes'],
        ['Tops', 'Skirts', 'Heels'],
    ],
    
    'Belts': [
        ['Shirts', 'Trousers', 'Formal Shoes'],
        ['Shirts', 'Jeans', 'Casual Shoes'],
        ['Tshirts', 'Jeans', 'Casual Shoes'],
    ],
    
    'Caps': [
        ['Tshirts', 'Shorts', 'Sports Shoes'],
        ['Sweatshirts', 'Track Pants', 'Sports Shoes'],
        ['Tshirts', 'Capris', 'Sports Shoes'],  # Women
    ],
    
    'Backpacks': [
        ['Tshirts', 'Jeans', 'Casual Shoes'],
        ['Sweatshirts', 'Trousers', 'Sports Shoes'],
        ['Shirts', 'Jeans', 'Casual Shoes'],
    ],
}

# NOTES:
# - Mỗi rule có 3-4 items (bao gồm payload product khi build outfit)
# - Cấu trúc: [Accessory, Bottom, Shoes] hoặc [Top, Accessory, Shoes] (cho Bottoms)
# - Dresses không cần Bottom nên chỉ có [Accessory, Shoes]
# - Women items được đánh dấu bằng comment để dễ phân biệt
# - Tất cả rules đều đảm bảo gender compatibility sẽ được check bởi logic
