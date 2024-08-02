# import pandas as pd
# from transformers import BertModel, BertTokenizer
# from sklearn.metrics.pairwise import cosine_similarity
# from flask import Flask, request, jsonify
# import torch
# import numpy as np

# # Đọc dữ liệu từ file CSV
# df = pd.read_csv('data.csv')

# # Kiểm tra dữ liệu
# print(df.head())

# # Loại bỏ các hàng có giá trị NaN trong cột 'full_description'
# df = df.dropna(subset=['full_description'])

# # Load pre-trained model and tokenizer
# model = BertModel.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def embed_texts(texts):
#     inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1)

# # Chuyển đổi các đoạn văn thành vector
# descriptions = df['full_description'].tolist()
# vectors = embed_texts(descriptions)

# def find_similar_vectors(input_vector, vectors, top_k=10):
#     similarities = cosine_similarity(input_vector, vectors)
#     top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
#     return top_k_indices, similarities[0][top_k_indices]

# # Xây dựng API với Flask
# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     short_description = data['description']
#     input_vector = embed_texts([short_description])
#     top_k_indices, similarities = find_similar_vectors(input_vector, vectors)

#     # Chỉ lấy những kết quả có điểm tương đồng trên 70%
#     threshold = 0.7
#     filtered_results = [
#         {"similarity": float(sim), "description": descriptions[idx]}
#         for idx, sim in zip(top_k_indices, similarities) if sim > threshold
#     ]

#     return jsonify(filtered_results)

# if __name__ == '__main__':
#     app.run(debug=True)
import pandas as pd
from sentence_transformers import SentenceTransformer, losses, util, InputExample
from torch.utils.data import DataLoader
import torch
import numpy as np

df = pd.read_csv('/kaggle/input/dataset/animals.csv')
df = df.dropna(subset=['full_description'])

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

train_examples = []
for _, row in df.iterrows():
    train_examples.append(InputExample(texts=[row['full_description'], row['full_description']], label=1.0))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=50, warmup_steps=100)

descriptions = df['full_description'].tolist()
vectors = model.encode(descriptions, convert_to_tensor=True)

def find_similar_vectors(input_vector, vectors, top_k=4):
    similarities = util.pytorch_cos_sim(input_vector, vectors)[0]
    similarities_cpu = similarities.cpu().numpy()
    top_k_indices = np.argsort(-similarities_cpu)[:top_k]
    return top_k_indices, similarities_cpu[top_k_indices]

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

if __name__ == '__main__':
    main()
