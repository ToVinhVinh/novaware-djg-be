# API Documentation - User Interactions

Tài liệu này mô tả các API liên quan đến việc quản lý interaction_history của user (view, like, cart, purchase, review).

## Base URL
```
/api/v1
```

---

## 1. Tạo Interaction Mới (Tự động cập nhật interaction_history)

### Endpoint
```
POST /api/v1/user-interactions
```

### Mô tả
Tạo một interaction mới trong collection `user_interactions` và tự động cập nhật `interaction_history` của user trong MongoDB.

**Lưu ý:** Nếu đã có interaction với `product_id` đó trong `interaction_history`, hệ thống sẽ **cập nhật** thay vì thêm mới.

### Request Body
```json
{
  "user_id": "691cd83e5fee2d4ca1ba6a46",
  "product_id": "10793",
  "interaction_type": "view",
  "rating": 5  // optional, chỉ dùng cho review
}
```

### Request Parameters
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | ID của user (ObjectId) |
| `product_id` | string/int | Yes | ID của product |
| `interaction_type` | string | Yes | Loại interaction: `view`, `like`, `cart`, `purchase`, `review` |
| `rating` | integer | No | Đánh giá (1-5), chỉ dùng khi `interaction_type` = `review` |

### Response Success (201 Created)
```json
{
  "success": true,
  "message": "User interaction created successfully",
  "data": {
    "interaction": {
      "id": "67890abcdef1234567890",
      "product_id": "10793",
      "interaction_type": "view",
      "rating": null,
      "timestamp": "2025-11-29T19:44:17.071000"
    }
  }
}
```

### Response Error (400 Bad Request)
```json
{
  "success": false,
  "message": "user_id is required when not logged in.",
  "data": null
}
```

### Ví dụ sử dụng

#### JavaScript/TypeScript (Fetch API)
```javascript
const createInteraction = async (userId, productId, interactionType) => {
  const response = await fetch('/api/v1/user-interactions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_id: userId,
      product_id: productId,
      interaction_type: interactionType
    })
  });
  
  const data = await response.json();
  return data;
};

// Sử dụng
await createInteraction('691cd83e5fee2d4ca1ba6a46', '10793', 'view');
```

#### Axios
```javascript
import axios from 'axios';

const createInteraction = async (userId, productId, interactionType) => {
  try {
    const response = await axios.post('/api/v1/user-interactions', {
      user_id: userId,
      product_id: productId,
      interaction_type: interactionType
    });
    return response.data;
  } catch (error) {
    console.error('Error creating interaction:', error.response?.data);
    throw error;
  }
};
```

---

## 2. Thêm Interaction vào interaction_history

### Endpoint
```
POST /api/v1/users/{user_id}/add_interaction
```

### Mô tả
Thêm một interaction trực tiếp vào `interaction_history` của user mà không tạo record trong collection `user_interactions`.

**Lưu ý:** Nếu đã có interaction với `product_id` đó, sẽ **thêm mới** (không cập nhật).

### URL Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | ID của user (ObjectId) |

### Request Body
```json
{
  "product_id": 10793,
  "interaction_type": "view",
  "timestamp": "2025-11-29T19:44:17.071000"  // optional
}
```

### Request Parameters
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `product_id` | integer/string | Yes | ID của product |
| `interaction_type` | string | Yes | Loại interaction: `view`, `like`, `cart`, `purchase`, `review` |
| `timestamp` | string | No | ISO format datetime string. Nếu không có, mặc định là thời gian hiện tại |

### Response Success (201 Created)
```json
{
  "success": true,
  "message": "Interaction added to user history successfully",
  "data": {
    "user_id": "691cd83e5fee2d4ca1ba6a46",
    "interaction": {
      "product_id": 10793,
      "interaction_type": "view",
      "timestamp": "2025-11-29T19:44:17.071000"
    },
    "total_interactions": 10
  }
}
```

### Response Error (400 Bad Request)
```json
{
  "success": false,
  "message": "product_id is required.",
  "data": null
}
```

### Response Error (404 Not Found)
```json
{
  "success": false,
  "message": "User does not exist.",
  "data": null
}
```

### Ví dụ sử dụng

#### JavaScript/TypeScript (Fetch API)
```javascript
const addInteraction = async (userId, productId, interactionType, timestamp = null) => {
  const body = {
    product_id: productId,
    interaction_type: interactionType
  };
  
  if (timestamp) {
    body.timestamp = timestamp;
  }
  
  const response = await fetch(`/api/v1/users/${userId}/add_interaction`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body)
  });
  
  const data = await response.json();
  return data;
};

// Sử dụng
await addInteraction('691cd83e5fee2d4ca1ba6a46', 10793, 'view');
```

---

## 3. Cập nhật Interaction Type (Quan trọng nhất)

### Endpoint
```
PUT /api/v1/users/{user_id}/update_interaction
PATCH /api/v1/users/{user_id}/update_interaction
```

### Mô tả
Cập nhật `interaction_type` cho một product cụ thể trong `interaction_history` của user. 

**Đây là API chính để cập nhật interaction_type** (ví dụ: từ "view" → "purchase").

**Lưu ý:** 
- Nếu đã có interaction với `product_id` đó, sẽ **cập nhật** interaction_type và timestamp
- Nếu chưa có, sẽ **tạo mới** interaction

### URL Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | ID của user (ObjectId) |

### Request Body
```json
{
  "product_id": 10793,
  "interaction_type": "purchase",
  "timestamp": "2025-11-29T20:00:00"  // optional
}
```

### Request Parameters
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `product_id` | integer/string | Yes | ID của product |
| `interaction_type` | string | Yes | Loại interaction mới: `view`, `like`, `cart`, `purchase`, `review` |
| `timestamp` | string | No | ISO format datetime string. Nếu không có, mặc định là thời gian hiện tại |

### Response Success (200 OK)
```json
{
  "success": true,
  "message": "Interaction updated successfully",
  "data": {
    "user_id": "691cd83e5fee2d4ca1ba6a46",
    "product_id": 10793,
    "interaction_type": "purchase",
    "updated": true,  // true nếu đã cập nhật, false nếu tạo mới
    "total_interactions": 10
  }
}
```

### Response Error (400 Bad Request)
```json
{
  "success": false,
  "message": "interaction_type must be one of: view, like, cart, purchase, review",
  "data": null
}
```

### Response Error (404 Not Found)
```json
{
  "success": false,
  "message": "User does not exist.",
  "data": null
}
```

### Ví dụ sử dụng

#### JavaScript/TypeScript (Fetch API)
```javascript
const updateInteraction = async (userId, productId, interactionType, timestamp = null) => {
  const body = {
    product_id: productId,
    interaction_type: interactionType
  };
  
  if (timestamp) {
    body.timestamp = timestamp;
  }
  
  const response = await fetch(`/api/v1/users/${userId}/update_interaction`, {
    method: 'PUT',  // hoặc 'PATCH'
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body)
  });
  
  const data = await response.json();
  return data;
};

// Ví dụ: User xem product → sau đó mua
await updateInteraction('691cd83e5fee2d4ca1ba6a46', 10793, 'view');
// ... sau đó user mua
await updateInteraction('691cd83e5fee2d4ca1ba6a46', 10793, 'purchase');
```

#### Axios
```javascript
import axios from 'axios';

const updateInteraction = async (userId, productId, interactionType) => {
  try {
    const response = await axios.put(`/api/v1/users/${userId}/update_interaction`, {
      product_id: productId,
      interaction_type: interactionType
    });
    return response.data;
  } catch (error) {
    console.error('Error updating interaction:', error.response?.data);
    throw error;
  }
};
```

---

## 4. Lấy danh sách Interactions

### Endpoint
```
GET /api/v1/user-interactions
```

### Mô tả
Lấy danh sách tất cả interactions từ collection `user_interactions`.

### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page` | integer | No | Số trang (mặc định: 1) |
| `page_size` | integer | No | Số items mỗi trang (mặc định: 20) |

### Response Success (200 OK)
```json
{
  "success": true,
  "message": "User interactions retrieved successfully",
  "data": {
    "interactions": [
      {
        "id": "67890abcdef1234567890",
        "product_id": "10793",
        "interaction_type": "view",
        "rating": null,
        "timestamp": "2025-11-29T19:44:17.071000"
      }
    ],
    "page": 1,
    "pages": 5,
    "perPage": 20,
    "count": 100
  }
}
```

---

## 5. Lấy thông tin User (bao gồm interaction_history)

### Endpoint
```
GET /api/v1/users/{user_id}
```

### Mô tả
Lấy thông tin chi tiết của user, bao gồm `interaction_history`.

### Response Success (200 OK)
```json
{
  "success": true,
  "message": "User retrieved successfully",
  "data": {
    "user": {
      "id": "691cd83e5fee2d4ca1ba6a46",
      "email": "user@example.com",
      "name": "User Name",
      "interaction_history": [
        {
          "product_id": 10793,
          "interaction_type": "purchase",
          "timestamp": "2025-11-29T19:44:17.071000"
        }
      ]
    }
  }
}
```

---

## Interaction Types

Các loại interaction được hỗ trợ:

| Type | Mô tả | Thứ tự ưu tiên |
|------|-------|----------------|
| `view` | User xem sản phẩm | 1 (thấp nhất) |
| `like` | User thích sản phẩm | 2 |
| `cart` | User thêm vào giỏ hàng | 3 |
| `purchase` | User mua sản phẩm | 4 (cao nhất) |
| `review` | User đánh giá sản phẩm | - |

**Lưu ý:** Khi cập nhật interaction_type, interaction cũ sẽ được thay thế hoàn toàn (không giữ lịch sử).

---

## Use Cases

### Use Case 1: User xem sản phẩm → Mua sản phẩm

```javascript
// Bước 1: User xem sản phẩm
await updateInteraction(userId, productId, 'view');

// Bước 2: User mua sản phẩm (tự động cập nhật từ 'view' → 'purchase')
await updateInteraction(userId, productId, 'purchase');
```

### Use Case 2: User thêm vào giỏ hàng → Mua

```javascript
// Bước 1: User thêm vào giỏ hàng
await updateInteraction(userId, productId, 'cart');

// Bước 2: User mua
await updateInteraction(userId, productId, 'purchase');
```

### Use Case 3: Tạo interaction và tự động cập nhật history

```javascript
// Tạo interaction mới (tự động cập nhật interaction_history)
await createInteraction(userId, productId, 'view');

// Nếu user mua sau đó, cập nhật lại
await updateInteraction(userId, productId, 'purchase');
```

---

## Error Handling

Tất cả các API đều trả về format chuẩn:

### Success Response
```json
{
  "success": true,
  "message": "Success message",
  "data": { ... }
}
```

### Error Response
```json
{
  "success": false,
  "message": "Error message",
  "data": null
}
```

### HTTP Status Codes
- `200 OK` - Thành công (GET, PUT, PATCH)
- `201 Created` - Tạo mới thành công (POST)
- `400 Bad Request` - Dữ liệu không hợp lệ
- `404 Not Found` - Không tìm thấy resource
- `500 Internal Server Error` - Lỗi server


