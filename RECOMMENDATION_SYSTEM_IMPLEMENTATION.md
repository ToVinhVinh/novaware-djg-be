# Product Recommendation System - Complete Implementation

## Overview

This document describes the complete implementation of a product recommendation system for the Django + MongoDB e-commerce platform. The system implements three distinct recommendation models with proper filtering, outfit logic, and Vietnamese reasons.

## Tech Stack

- **Backend**: Django + Django REST Framework
- **Database**: MongoDB (via mongoengine)
- **Models**:
  1. **GNN**: PyTorch Geometric + LightGCN
  2. **Content-Based**: Sentence-BERT (all-MiniLM-L6-v2) + FAISS
  3. **Hybrid**: Late fusion (0.7 GNN + 0.3 Content)

## API Endpoints

### 1. GNN (Graph Neural Network) Recommendations

#### Train Model
```
POST /api/v1/recommend/gnn/train/
```

**Request Body**:
```json
{
  "force_retrain": false,
  "sync": true
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Model trained successfully",
  "num_users": 1000,
  "num_products": 5000,
  "num_interactions": 10000,
  "embedding_dim": 64,
  "num_layers": 3,
  "trained_at": "2025-11-17T10:38:39.835Z"
}
```

#### Get Recommendations
```
GET /api/v1/recommend/gnn/recommend/?user_id=<user_id>&product_id=<product_id>
```

**Query Parameters**:
- `user_id` (required): MongoDB ObjectId of the user
- `product_id` (required): MongoDB ObjectId of the current product
- `top_k_personal` (optional, default=5): Number of personalized recommendations
- `top_k_outfit` (optional, default=4): Number of outfit items per category

**Response**:
```json
{
  "personalized": [
    {
      "product": {
        "id": "690bf0f2d0c3753df0ecbdd6",
        "name": "Turtle Check Men Navy Blue Shirt",
        "gender": "Men",
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": "Shirts",
        "baseColour": "Navy Blue",
        "season": "Fall",
        "usage": "Casual",
        "images": ["https://..."]
      },
      "score": 0.8765,
      "reason": "Dựa trên độ tuổi 27 và giới tính nam, bạn từng tương tác nhiều với áo sơ mi casual, màu navy blue là màu bạn yêu thích"
    }
  ],
  "outfit": {
    "bottoms": {
      "product": {...},
      "score": 0.7543,
      "reason": "Phối hợp hoàn hảo với áo sơ mi: quần jeans slim, màu xanh đậm hài hòa với Navy Blue"
    },
    "shoes": {...},
    "accessories": {...}
  },
  "reasons": {
    "personalized": [
      "Dựa trên độ tuổi 27 và giới tính nam...",
      "..."
    ],
    "outfit": [
      "Phối hợp hoàn hảo với áo sơ mi"
    ]
  }
}
```

### 2. Content-Based Recommendations

#### Train Model
```
POST /api/v1/recommend/content/train/
```

#### Get Recommendations
```
GET /api/v1/recommend/content/recommend/?user_id=<user_id>&product_id=<product_id>
```

Response format is identical to GNN endpoint.

### 3. Hybrid Recommendations

#### Train Model
```
POST /api/v1/recommend/hybrid/train/
```

#### Get Recommendations
```
GET /api/v1/recommend/hybrid/recommend/?user_id=<user_id>&product_id=<product_id>&alpha=0.7
```

**Additional Query Parameter**:
- `alpha` (optional, default=0.7): Weight for GNN component (CBF gets 1-alpha)

## Core Features

### 1. Strict Filtering Rules

All recommendations are filtered by:

- **Gender**: Never recommend opposite gender items
  - Male users: Only "Men" or "Unisex" products
  - Female users: Only "Women" or "Unisex" products

- **Age**: Filter inappropriate age groups
  - Skip kids items for adults (age >= 18)

- **Deduplication**: Remove exact matches and very similar items (similarity >= 0.95)

### 2. Outfit Logic

Outfit recommendations follow strict category rules:

| Current Product | Outfit Categories |
|----------------|-------------------|
| Topwear (tops) | Bottomwear + Shoes + Accessories |
| Bottomwear (bottoms) | Topwear + Shoes + Accessories |
| Dress | Shoes + Accessories only |
| Shoes / Accessories | Topwear + Bottomwear (+ Dresses for female) |

**Important Rules**:
- Never recommend dresses when viewing tops/bottoms
- Dresses only for female users
- Each outfit category gets exactly 1 product

### 3. Vietnamese Reasons

All recommendations include specific Vietnamese reasons mentioning:

- **Age**: "Dựa trên độ tuổi 27"
- **Gender**: "giới tính nam/nữ"
- **Interaction History**: "bạn từng tương tác nhiều với áo sơ mi casual"
- **Style Preferences**: "phong cách casual bạn thường chọn"
- **Color Preferences**: "màu navy blue là màu bạn yêu thích"

Example reasons:
```
"Dựa trên độ tuổi 27 và giới tính nam, bạn từng tương tác nhiều với áo sơ mi casual màu navy, phong cách casual bạn thường chọn"

"Phối hợp hoàn hảo với áo sơ mi: quần jeans slim + giày sneakers trắng + belt da"
```

## Implementation Details

### 1. GNN Model (LightGCN)

**File**: `apps/recommendations/gnn/engine_lightgcn.py`

**Architecture**:
- Bipartite graph: User ↔ Product
- 3 LightGCN layers
- 64-dimensional embeddings
- BPR (Bayesian Personalized Ranking) loss
- Message passing without feature transformation

**Training**:
- Loads all user-product interactions
- Builds bipartite graph with edge weights based on interaction type
- Trains for 50 epochs with Adam optimizer
- Saves model to `models/gnn_lightgcn.pkl`

**Recommendation**:
- Computes user-product scores using learned embeddings
- Filters by age/gender
- Applies outfit logic
- Generates Vietnamese reasons

### 2. Content-Based Model (Sentence-BERT + FAISS)

**File**: `apps/recommendations/cbf/engine_sbert_faiss.py`

**Architecture**:
- Sentence-BERT model: `all-MiniLM-L6-v2`
- FAISS index: IVF Flat (for >10k products) or Flat L2 (for smaller datasets)
- Product embeddings: Generated from text features (name, category, color, usage, season)
- User profiles: Weighted average of interacted product embeddings

**Training**:
- Generates embeddings for all products using Sentence-BERT
- Builds FAISS index for efficient similarity search
- Creates user profiles from interaction history
- Saves model to `models/cbf_sbert_faiss.pkl`

**Recommendation**:
- Searches FAISS index for similar products
- Filters by age/gender
- Applies outfit logic
- Generates Vietnamese reasons

### 3. Hybrid Model (Late Fusion)

**File**: `apps/recommendations/hybrid/engine_fusion.py`

**Architecture**:
- Combines GNN and CBF scores using weighted sum
- Default weights: 0.7 GNN (CF) + 0.3 CBF (Content)
- Late fusion: Scores are combined after both models generate recommendations

**Training**:
- Trains both GNN and CBF models
- No additional training required

**Recommendation**:
- Gets recommendations from both GNN and CBF
- Combines scores: `final_score = alpha * gnn_score + (1 - alpha) * cbf_score`
- Re-ranks products by combined score
- Filters by age/gender
- Applies outfit logic
- Generates Vietnamese reasons with fusion info

## Utility Modules

### 1. Category Mapper

**File**: `apps/recommendations/utils/category_mapper.py`

Maps product categories to standardized tags:
- Topwear → "tops"
- Bottomwear → "bottoms"
- Dress → "dresses"
- Shoes/Flip Flops/Sandal → "shoes"
- Bags/Belts/Headwear/Watches → "accessories"

### 2. Filters

**File**: `apps/recommendations/utils/filters.py`

Functions:
- `filter_by_age_gender()`: Filter products by user demographics
- `get_outfit_categories()`: Get complementary outfit categories
- `deduplicate_products()`: Remove duplicate/very similar products
- `normalize_gender()`: Normalize gender strings

### 3. Embedding Generator

**File**: `apps/recommendations/utils/embedding_generator.py`

Functions:
- `generate_embedding()`: Generate embedding for single product
- `generate_embeddings_batch()`: Batch embedding generation
- `generate_user_embedding()`: Create user profile from interactions
- `compute_similarity()`: Compute cosine similarity

### 4. Reason Generator

**File**: `apps/recommendations/utils/reasons.py`

Functions:
- `generate_vietnamese_reason()`: Generate personalized Vietnamese reasons
- `generate_outfit_reason()`: Generate outfit-specific reasons
- `_translate_article_type()`: Translate product types to Vietnamese

## Data Models

### User Document (MongoDB)

```python
{
    "id": "690bf0f2d0c3753df0ecbdd6",
    "email": "user@example.com",
    "name": "Nguyen Thanh Doanh",
    "gender": "male",
    "age": 27,
    "preferences": {
        "priceRange": {"min": 0, "max": 1000000},
        "style": "casual",
        "colorPreferences": ["navy", "black"],
        "brandPreferences": []
    },
    "interaction_history": [
        {
            "product_id": "...",
            "interaction_type": "purchase",
            "timestamp": "..."
        }
    ]
}
```

### Product Document (MongoDB)

```python
{
    "id": 15970,
    "gender": "Men",
    "masterCategory": "Apparel",
    "subCategory": "Topwear",
    "articleType": "Shirts",
    "baseColour": "Navy Blue",
    "season": "Fall",
    "year": 2011,
    "usage": "Casual",
    "productDisplayName": "Turtle Check Men Navy Blue Shirt",
    "images": ["https://..."],
    "rating": 5.0,
    "variants": [
        {
            "stock": 15,
            "color": "#000000",
            "size": "m",
            "price": 346837.0
        }
    ]
}
```

## Model Persistence

All models are saved to the `models/` directory:

```
models/
├── gnn_lightgcn.pkl          # GNN model + embeddings
├── cbf_sbert_faiss.pkl       # CBF product embeddings + user profiles
├── cbf_faiss.index           # FAISS index
```

## Background Training

Training can be run in two modes:

1. **Synchronous** (`sync=true`): Blocks until training completes
2. **Asynchronous** (`sync=false`): Returns task ID immediately, check status later

## Testing

### Test GNN Endpoint

```bash
# Train
curl -X POST http://localhost:8000/api/v1/recommend/gnn/train/ \
  -H "Content-Type: application/json" \
  -d '{"force_retrain": true, "sync": true}'

# Recommend
curl "http://localhost:8000/api/v1/recommend/gnn/recommend/?user_id=690bf0f2d0c3753df0ecbdd6&product_id=15970"
```

### Test Content Endpoint

```bash
# Train
curl -X POST http://localhost:8000/api/v1/recommend/content/train/ \
  -H "Content-Type: application/json" \
  -d '{"force_retrain": true, "sync": true}'

# Recommend
curl "http://localhost:8000/api/v1/recommend/content/recommend/?user_id=690bf0f2d0c3753df0ecbdd6&product_id=15970"
```

### Test Hybrid Endpoint

```bash
# Train
curl -X POST http://localhost:8000/api/v1/recommend/hybrid/train/ \
  -H "Content-Type: application/json" \
  -d '{"force_retrain": true, "sync": true}'

# Recommend (with custom alpha)
curl "http://localhost:8000/api/v1/recommend/hybrid/recommend/?user_id=690bf0f2d0c3753df0ecbdd6&product_id=15970&alpha=0.7"
```

## Performance Considerations

1. **FAISS Index**: Uses IVF Flat for datasets > 10k products for faster search
2. **Caching**: GNN co-occurrence graph is cached in memory with TTL
3. **Batch Processing**: Embeddings are generated in batches for efficiency
4. **Interaction Lookback**: Only processes last 90 days of interactions for GNN training

## Error Handling

All endpoints handle common errors:

- **Model Not Trained**: Returns 409 Conflict with message to train first
- **User/Product Not Found**: Returns 400 Bad Request
- **Invalid Parameters**: Returns 400 Bad Request with validation errors
- **Training Failures**: Returns 500 Internal Server Error with error details

## Future Enhancements

1. **Image Embeddings**: Add CLIP for visual similarity
2. **Real-time Updates**: Incremental model updates without full retraining
3. **A/B Testing**: Framework for testing different fusion weights
4. **Explainability**: More detailed reason generation with feature importance
5. **Cold Start**: Better handling of new users/products
6. **Diversity**: Add diversity metrics to avoid filter bubbles

## Conclusion

This implementation provides a complete, production-ready recommendation system with:
- ✅ 3 distinct models (GNN, Content, Hybrid)
- ✅ 6 API endpoints (3 train + 3 recommend)
- ✅ Strict filtering by age, gender, outfit logic
- ✅ Vietnamese reasons with specific user context
- ✅ Efficient storage and retrieval
- ✅ Background training support
- ✅ Comprehensive error handling

The system is ready for deployment and can be easily extended with additional features.

