# Hướng dẫn Training và Kiểm tra Status

## Cách 1: Sử dụng API trực tiếp

### Bước 1: Bắt đầu Training

**Gửi request:**
```bash
POST http://localhost:8000/api/v1/gnn/train
Content-Type: application/json

{
  "force_retrain": false
}
```

**Hoặc dùng curl:**
```bash
curl -X POST http://localhost:8000/api/v1/gnn/train \
  -H "Content-Type: application/json" \
  -d '{"force_retrain": false}'
```

**Response:**
```json
{
  "task_id": "abc-123-def-456",
  "model": "gnn",
  "status": "pending",
  "message": "Task is waiting to be processed",
  "progress": 0
}
```

**Lưu lại `task_id` để kiểm tra status!**

---

### Bước 2: Kiểm tra Status

**Gửi request với task_id:**
```bash
POST http://localhost:8000/api/v1/gnn/train
Content-Type: application/json

{
  "task_id": "abc-123-def-456"
}
```

**Response khi đang chạy:**
```json
{
  "task_id": "abc-123-def-456",
  "model": "gnn",
  "status": "running",
  "message": "Training in progress",
  "progress": 50,
  "current_step": "Processing data...",
  "total_steps": "4"
}
```

**Response khi hoàn thành:**
```json
{
  "task_id": "abc-123-def-456",
  "model": "gnn",
  "status": "success",
  "message": "Training completed successfully",
  "progress": 100,
  "result": {
    "status": "training_completed",
    "model": "gnn"
  },
  "training_info": {
    "trained_at": "2024-01-01T12:00:00",
    "model_name": "gnn"
  },
  "metrics": {...},
  "matrix_data": {...}
}
```

**Response khi thất bại:**
```json
{
  "task_id": "abc-123-def-456",
  "model": "gnn",
  "status": "failure",
  "message": "Training failed",
  "error": "Error message here",
  "progress": 0
}
```

---

## Cách 2: Sử dụng Python Script

### Chạy script tự động:

```bash
python test_training.py
```

Script sẽ:
1. ✅ Bắt đầu training
2. ✅ Tự động polling status mỗi 2 giây
3. ✅ Hiển thị progress real-time
4. ✅ Báo khi training hoàn thành

### Tùy chỉnh script:

Mở file `test_training.py` và thay đổi:
- `MODEL = "gnn"` → Đổi thành `"cbf"` hoặc `"hybrid"`
- `BASE_URL = "http://localhost:8000"` → Đổi URL nếu cần
- `check_interval=2` → Thay đổi thời gian kiểm tra (giây)
- `max_wait=300` → Thay đổi timeout (giây)

---

## Cách 3: Sử dụng JavaScript/Frontend

### Ví dụ với Fetch API:

```javascript
// Bước 1: Bắt đầu training
async function startTraining() {
  const response = await fetch('http://localhost:8000/api/v1/gnn/train', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      force_retrain: false
    })
  });
  
  const data = await response.json();
  console.log('Training started:', data);
  return data.task_id;
}

// Bước 2: Kiểm tra status (polling)
async function checkStatus(taskId) {
  const response = await fetch('http://localhost:8000/api/v1/gnn/train', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      task_id: taskId
    })
  });
  
  const data = await response.json();
  return data;
}

// Bước 3: Đợi training hoàn thành
async function waitForCompletion(taskId) {
  const checkInterval = 2000; // 2 giây
  const maxWait = 300000; // 5 phút
  
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    
    const check = async () => {
      const elapsed = Date.now() - startTime;
      
      if (elapsed > maxWait) {
        reject(new Error('Timeout'));
        return;
      }
      
      const status = await checkStatus(taskId);
      
      // Update UI với progress
      updateProgressBar(status.progress);
      updateStatusMessage(status.message);
      
      if (status.status === 'success') {
        resolve(status);
      } else if (status.status === 'failure') {
        reject(new Error(status.error));
      } else {
        // Tiếp tục kiểm tra
        setTimeout(check, checkInterval);
      }
    };
    
    check();
  });
}

// Sử dụng
async function trainModel() {
  try {
    const taskId = await startTraining();
    const result = await waitForCompletion(taskId);
    console.log('Training completed!', result);
  } catch (error) {
    console.error('Training failed:', error);
  }
}
```

---

## Cách 4: Sử dụng Postman/Thunder Client

### Collection cho Postman:

1. **Start Training:**
   - Method: `POST`
   - URL: `http://localhost:8000/api/v1/gnn/train`
   - Body (JSON):
     ```json
     {
       "force_retrain": false
     }
     ```

2. **Check Status:**
   - Method: `POST`
   - URL: `http://localhost:8000/api/v1/gnn/train`
   - Body (JSON):
     ```json
     {
       "task_id": "{{task_id}}"
     }
     ```
   - Lưu `task_id` từ response của request 1 vào variable `{{task_id}}`

---

## Các Status có thể có:

| Status | Ý nghĩa | Hành động |
|--------|---------|-----------|
| `pending` | Task đang chờ được xử lý | Tiếp tục kiểm tra |
| `running` | Training đang chạy | Tiếp tục kiểm tra, hiển thị progress |
| `success` | Training hoàn thành ✅ | Dừng kiểm tra, hiển thị kết quả |
| `failure` | Training thất bại ❌ | Dừng kiểm tra, hiển thị lỗi |
| `not_found` | Task không tồn tại | Task đã bị xóa hoặc không hợp lệ |

---

## Tips:

1. **Polling Interval:** Kiểm tra mỗi 2-5 giây là hợp lý
2. **Timeout:** Đặt timeout 5-10 phút tùy vào độ phức tạp của model
3. **UI Feedback:** Hiển thị progress bar và status message cho user
4. **Error Handling:** Luôn xử lý trường hợp `failure` và `not_found`

---

## Training đồng bộ (Sync Mode)

Nếu muốn chờ kết quả ngay (không cần polling):

```json
POST http://localhost:8000/api/v1/gnn/train
{
  "force_retrain": false,
  "sync": true
}
```

Response sẽ trả về kết quả ngay khi training hoàn thành (có thể mất vài phút).

---

## Các Models khác:

- **CBF:** `/api/v1/cbf/train/`
- **Hybrid:** `/api/v1/hybrid/train/`

Cách sử dụng tương tự, chỉ cần thay URL.

