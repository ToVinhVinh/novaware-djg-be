import pandas as pd

df = pd.read_csv('apps/exports/products.csv')

print("Checking Jeans availability for Women/Unisex:")
jeans = df[(df['articleType'] == 'Jeans') & ((df['gender'] == 'Women') | (df['gender'] == 'Unisex'))]
print(f"Total: {len(jeans)} products")

if len(jeans) > 0:
    print("\nFirst 5:")
    print(jeans[['id', 'articleType', 'gender', 'productDisplayName']].head())
else:
    print("\n❌ NO JEANS FOR WOMEN/UNISEX!")
    print("\nChecking all Jeans:")
    all_jeans = df[df['articleType'] == 'Jeans']
    print(f"Total Jeans: {len(all_jeans)}")
    print(all_jeans['gender'].value_counts())
