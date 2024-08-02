
# Sentence Similarity Finder

## Giới thiệu

Cái nì gọi là "Sentence Similarity Finder" sử dụng mô hình `SentenceTransformer` để tìm kiếm những đoạn văn bản tương tự trong bộ dữ liệu mô tả động vật. 🐾
thì đề bài là nhập mô tả về một con vật thì nó sẽ tìm trong data và trả ra các đoạn văn bản có liên quan.
cái này nó rất là cơ bản, data rất là ít nên là kết quả trả ra ko có chính xác đc 100% đâu nên là chặt chém a nha =)))
## Nếu dùng Kaggle
thì chỉ cần copy code dán lên, chọn máy có GPU, cài thư viện, tải file csv lên là đc nha
## Cài đặt

### Bước 1: Chuẩn bị Python

Đảm bảo bạn đã cài đặt Python (>= 3.9.6) trên máy tính của mình. Nếu mn đang sử dụng Kaggle thì không cần lo lắng về bước này.

### Bước 2: Tạo môi trường ảo

Chúng ta sẽ tạo một môi trường ảo để cài đặt các thư viện cần thiết:
```sh
python -m venv venv
source venv/bin/activate  # Trên Windows, dùng `venv\Scripts\activate`
```

### Bước 3: Cài đặt thư viện

Cài đặt các thư viện cần thiết. Nếu gặp lỗi thì hãy gáng thử lại với các phiên bản khác nhau nha hehe:
```sh
pip install pandas sentence-transformers torch numpy
```

## Hướng dẫn sử dụng (đọc kĩ trước khi sử dụng nha 😄)

### Bước 1: Chuẩn bị dữ liệu

Đặt file `animals.csv` vào thư mục phù hợp (ví dụ: `/kaggle/input/dataset/animals.csv`). Đường dẫn có thể tùy chỉnh theo ý mn miễn là khớp với vị trí đặt file.

### Bước 2: Chạy chương trình

Chạy file `main.py` để bắt đầu chương trình:
```sh
python main.py
```

### Bước 3: Nhập mô tả ngắn

Nhập mô tả ngắn khi được yêu cầu. Chương trình sẽ tìm và trả về các mô tả tương tự từ bộ dữ liệu nếu có.

## Giải thích chi tiết

### Đọc dữ liệu

Mã này sẽ đọc dữ liệu từ file CSV và loại bỏ các hàng có giá trị NaN trong cột `full_description`:
```python
df = pd.read_csv('/kaggle/input/dataset/animals.csv')
df = df.dropna(subset=['full_description'])
```

### Load mô hình

Tải mô hình `SentenceTransformer` để sử dụng cho việc chuyển đổi văn bản thành vector:
```python
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
```

### Chuẩn bị dữ liệu và Fine-tuning

1. Chuẩn bị dữ liệu dưới dạng `InputExample`:
   ```python
   train_examples = []
   for _, row in df.iterrows():
       train_examples.append(InputExample(texts=[row['full_description'], row['full_description']], label=1.0))
   ```

2. Đưa dữ liệu vào `DataLoader` để phục vụ quá trình fine-tuning:
   ```python
   train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
   train_loss = losses.CosineSimilarityLoss(model)
   ```

3. Fine-tune mô hình với `CosineSimilarityLoss` qua 50 epochs:
   ```python
   model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=50, warmup_steps=100)
   ```

### Chuyển đổi các đoạn văn thành vector

Chuyển các mô tả trong bộ dữ liệu thành vector:
```python
descriptions = df['full_description'].tolist()
vectors = model.encode(descriptions, convert_to_tensor=True)
```

### Hàm tìm vector tương tự

Hàm `find_similar_vectors` dùng để tìm các vector có điểm tương đồng cao:
```python
def find_similar_vectors(input_vector, vectors, top_k=4):
    similarities = util.pytorch_cos_sim(input_vector, vectors)[0]
    similarities_cpu = similarities.cpu().numpy()
    top_k_indices = np.argsort(-similarities_cpu)[:top_k]
    return top_k_indices, similarities_cpu[top_k_indices]
```

### Chương trình chính

Nhận mô tả ngắn từ người dùng, chuyển thành vector và tìm các vector tương tự:
```python
def main():
    while True:
        short_description = input("Nhập mô tả ngắn (hoặc 'exit' để thoát): ")
        if short_description.lower() == 'exit':
            break
        input_vector = model.encode([short_description], convert_to_tensor=True)
        top_k_indices, similarities = find_similar_vectors(input_vector, vectors)
        threshold = 0.7
        filtered_results = [
            {"similarity": float(sim), "description": descriptions[idx]}
            for idx, sim in zip(top_k_indices, similarities) if sim > threshold
        ]
        if filtered_results:
            print("Các mô tả tương tự:")
            for result in filtered_results:
                print(f"Điểm tương đồng: {result['similarity']:.2f} - Mô tả: {result['description']}")
        else:
            print("Không có mô tả tương tự nào đạt ngưỡng tương đồng.")
```

### Chạy chương trình

Gọi hàm `main` để chạy chương trình:
```python
if __name__ == '__main__':
    main()
```

## Liên hệ

Nếu có thắc mắc hay cần hỗ trợ, vui lòng liên hệ qua email: [pmkkhoaminh@gmail.com](mailto:pmkkhoaminh@gmail.com). Mình rất khó chệu vô cùng 💌
