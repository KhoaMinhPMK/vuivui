
# Sentence Similarity Finder

## Giới thiệu

cái nì sử dụng mô hình `SentenceTransformer` để tìm các đoạn văn bản tương tự từ một bộ dữ liệu các mô tả động vật.

## Cài đặt

1. Cài đặt Python (>= 3.9.6) nếu dùng trên laptop nha, còn dùng trên kaggle thì khỏi lo.
------ phần này dành cho ai dùng trên máy thôi----------------------------------------------------------------
2. Tạo môi trường ảo và kích hoạt:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
   cài thư viện mà có lỗi thì chịu khó mò phiên bản tí nhe hehe
   ```sh
   pip install pandas sentence-transformers torch numpy
   ```

## Hướng dẫn sử dụng(đọc kĩ trước khi sử dụng ạ)

1. Đặt file `animals.csv` vào thư mục phù hợp (ví dụ: `/kaggle/input/dataset/animals.csv`). đường dẫn thì tùy biến vô tư nha
2. Chạy file `main.py` để bắt đầu chương trình:
   ```sh
   python main.py
   ```
3. Nhập mô tả ngắn vào khi được yêu cầu. Chương trình sẽ trả về các mô tả tương tự từ bộ dữ liệu nếu tìm thấy.

## Giải thích chi tiết

### Đọc dữ liệu

Mã đọc dữ liệu từ file CSV và loại bỏ các hàng có giá trị NaN trong cột `full_description`:
```python
df = pd.read_csv('/kaggle/input/dataset/animals.csv')
df = df.dropna(subset=['full_description'])
```

### Load mô hình

Mô hình `SentenceTransformer` được tải để sử dụng cho việc chuyển đổi văn bản thành vector:
```python
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
```

### Chuẩn bị dữ liệu và Fine-tuning

1. Dữ liệu được chuẩn bị dưới dạng `InputExample`, mỗi `InputExample` chứa hai đoạn văn bản giống nhau và một nhãn (label) là 1.0:
   ```python
   train_examples = []
   for _, row in df.iterrows():
       train_examples.append(InputExample(texts=[row['full_description'], row['full_description']], label=1.0))
   ```

2. Dữ liệu được đưa vào `DataLoader` để phục vụ cho quá trình fine-tuning:
   ```python
   train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
   train_loss = losses.CosineSimilarityLoss(model)
   ```

3. Mô hình được fine-tune với `CosineSimilarityLoss` qua 50 epochs:
   ```python
   model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=50, warmup_steps=100)
   ```

### Chuyển đổi các đoạn văn thành vector

Các mô tả trong bộ dữ liệu được chuyển thành vector:
```python
descriptions = df['full_description'].tolist()
vectors = model.encode(descriptions, convert_to_tensor=True)
```

### Hàm tìm vector tương tự

Hàm `find_similar_vectors` được sử dụng để tìm các vector có điểm tương đồng cao với vector đầu vào:
```python
def find_similar_vectors(input_vector, vectors, top_k=4):
    similarities = util.pytorch_cos_sim(input_vector, vectors)[0]
    similarities_cpu = similarities.cpu().numpy()
    top_k_indices = np.argsort(-similarities_cpu)[:top_k]
    return top_k_indices, similarities_cpu[top_k_indices]
```

### Chương trình chính

Chương trình nhận mô tả ngắn từ người dùng, chuyển thành vector và tìm các vector tương tự:
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

Chương trình được chạy bằng cách gọi hàm `main` trong đoạn mã sau:
```python
if __name__ == '__main__':
    main()
```

## Liên hệ

Nếu có thắc mắc hay cần hỗ trợ, vui lòng liên hệ qua email: [pmkkhoaminh@gmail.com](mailto:pmkkhoaminh@gmail.com)
