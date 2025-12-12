"""
Complement dictionary for outfit suggestions.
Defines which item types are compatible with each other.
"""

# Item-Item complement dictionary
COMPLEMENT = {
    'Trousers': [['Tshirts', 'Shirts', 'Jackets', 'Sweaters', 'Sweatshirts', 'Formal Shoes'], ['Tshirts', 'Shirts', 'Jackets', 'Sweaters', 'Sweatshirts', 'Casual Shoes'], ['Tshirts', 'Shirts', 'Jackets', 'Sweaters', 'Sweatshirts', 'Sports Shoes']],
    'Tshirts': [['Watches', 'Jeans', 'Casual Shoes'], ['Watches', 'Jeans', 'Flip Flops']],
    'Shirts': [['Trousers', 'Formal Shoes', 'Watches'], ['Jeans', 'Formal Shoes', 'Watches'], ['Shorts', 'Formal Shoes', 'Watches'], ['Trousers', 'Casual Shoes', 'Watches'], ['Jeans', 'Casual Shoes', 'Watches'], ['Shorts', 'Casual Shoes', 'Watches'], ['Trousers', 'Formal Shoes', 'Belts'], ['Jeans', 'Formal Shoes', 'Belts'], ['Shorts', 'Formal Shoes', 'Belts'], ['Trousers', 'Casual Shoes', 'Belts'], ['Jeans', 'Casual Shoes', 'Belts'], ['Shorts', 'Casual Shoes', 'Belts'], ['Trousers', 'Formal Shoes', 'Watches', 'Belts'], ['Jeans', 'Formal Shoes', 'Watches', 'Belts'], ['Trousers', 'Casual Shoes', 'Watches', 'Belts'], ['Jeans', 'Casual Shoes', 'Watches', 'Belts']],
    'Dresses': [['Watches', 'Casual Shoes'], ['Watches', 'Flats'], ['Watches', 'Flip Flops']],
    'Tops': [['Trousers', 'Casual Shoes'], ['Jeans', 'Casual Shoes'], ['Shorts', 'Casual Shoes'], ['Skirts', 'Casual Shoes'], ['Capris', 'Casual Shoes'], ['Trousers', 'Sports Shoes'], ['Jeans', 'Sports Shoes'], ['Shorts', 'Sports Shoes'], ['Skirts', 'Sports Shoes'], ['Capris', 'Sports Shoes']],
    'Shorts': [['Tshirts', 'Sweatshirts', 'Sports Shoes'], ['Tops', 'Sweatshirts', 'Sports Shoes'], ['Tshirts', 'Sweatshirts', 'Casual Shoes'], ['Tops', 'Sweatshirts', 'Casual Shoes'], ['Tshirts', 'Sweatshirts', 'Flip Flops'], ['Tops', 'Sweatshirts', 'Flip Flops'], ['Watches', 'Tshirts', 'Sports Shoes']],
    'Skirts': [['Tshirts', 'Tunics', 'Jackets', 'Heels'], ['Tops', 'Tunics', 'Jackets', 'Heels'], ['Tshirts', 'Tunics', 'Jackets', 'Flats'], ['Tops', 'Tunics', 'Jackets', 'Flats'], ['Tshirts', 'Tunics', 'Jackets', 'Casual Shoes'], ['Tops', 'Tunics', 'Jackets', 'Casual Shoes'], ['Watches', 'Tshirts', 'Casual Shoes'], ['Watches', 'Tshirts', 'Flats'], ['Watches', 'Tshirts', 'Flip Flops']],
    'Jeans': [['Tshirts', 'Shirts', 'Sweaters', 'Sweatshirts', 'Jackets', 'Casual Shoes'], ['Tops', 'Shirts', 'Sweaters', 'Sweatshirts', 'Jackets', 'Casual Shoes'], ['Tshirts', 'Shirts', 'Sweaters', 'Sweatshirts', 'Jackets', 'Sports Shoes'], ['Tops', 'Shirts', 'Sweaters', 'Sweatshirts', 'Jackets', 'Sports Shoes'], ['Watches', 'Tshirts', 'Flip Flops'], ['Watches', 'Shirts', 'Casual Shoes']],
    'Formal Shoes': ['Watches', 'Shirts', 'Trousers'],
    'Casual Shoes': [['Watches', 'Tshirts', 'Jeans'], ['Watches', 'Shirts', 'Jeans']],
    'Sports Shoes': [['Tshirts', 'Shorts'], ['Tops', 'Shorts'], ['Tshirts', 'Track Pants'], ['Tops', 'Track Pants'], ['Tshirts', 'Capris'], ['Tops', 'Capris'], ['Watches', 'Tshirts', 'Shorts'], ['Watches', 'Tshirts', 'Track Pants']],
    'Heels': [['Watches', 'Tshirts', 'Skirts'], ['Watches', 'Dresses']],
    'Flats': [['Watches', 'Tshirts', 'Skirts', 'Dresses'], ['Watches', 'Tshirts', 'Shorts', 'Dresses']],
    'Sandals': [['Tshirts', 'Shorts'], ['Tops', 'Shorts'], ['Watches', 'Tshirts', 'Jeans'], ['Watches', 'Shirts', 'Jeans']],
    'Flip Flops': [['Watches', 'Tshirts', 'Jeans'], ['Watches', 'Shirts', 'Jeans']],
    'Handbags': [['Tshirts', 'Skirts', 'Casual Shoes'], ['Tshirts', 'Skirts', 'Flats'], ['Tshirts', 'Skirts', 'Flip Flops'], ['Dresses', 'Flip Flops'], ['Dresses', 'Flats']],
    'Jackets': [['Trousers', 'Tshirts', 'Dresses', 'Shirts'], ['Jeans', 'Tshirts', 'Dresses', 'Shirts'], ['Trousers', 'Tops', 'Dresses', 'Shirts'], ['Jeans', 'Tops', 'Dresses', 'Shirts']],
    'Sweaters': [['Trousers', 'Dresses'], ['Jeans', 'Dresses']],
    'Sweatshirts': [['Trousers', 'Casual Shoes', 'Watches'], ['Trousers', 'Formal Shoes', 'Watches'], ['Jeans', 'Sports Shoes', 'Watches'], ['Jeans', 'Casual Shoes', 'Watches'], ['Shorts', 'Sports Shoes', 'Watches'], ['Shorts', 'Flip Flops', 'Watches'], ['Track Pants', 'Sports Shoes', 'Watches'], ['Trousers', 'Casual Shoes', 'Caps'], ['Trousers', 'Formal Shoes', 'Caps'], ['Jeans', 'Sports Shoes', 'Caps'], ['Jeans', 'Casual Shoes', 'Caps'], ['Shorts', 'Sports Shoes', 'Caps'], ['Shorts', 'Flip Flops', 'Caps'], ['Track Pants', 'Sports Shoes', 'Caps'], ['Trousers', 'Casual Shoes', 'Backpacks'], ['Jeans', 'Sports Shoes', 'Backpacks'], ['Shorts', 'Sports Shoes', 'Backpacks'], ['Track Pants', 'Sports Shoes', 'Backpacks'], ['Trousers', 'Casual Shoes', 'Watches', 'Caps'], ['Trousers', 'Formal Shoes', 'Watches'], ['Jeans', 'Sports Shoes', 'Watches', 'Caps'], ['Shorts', 'Sports Shoes', 'Watches', 'Caps'], ['Track Pants', 'Sports Shoes', 'Watches', 'Caps']],
    'Backpacks': [['Tshirts', 'Jeans', 'Flip Flops'], ['Shirts', 'Jeans', 'Casual Shoes']],
    'Belts': [['Tshirts', 'Jeans', 'Flip Flops'], ['Shirts', 'Jeans', 'Casual Shoes']],
    'Capris': [['Caps', 'Jackets', 'Sports Shoes'], ['Caps', 'Tshirts', 'Sports Shoes']],
    'Caps': [['Tshirts', 'Shorts', 'Sports Shoes'], ['Tshirts', 'Track Pants', 'Sports Shoes']]
}

