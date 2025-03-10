# 1. Giới thiệu
## 1. Khái niệm cơ bản và Ý tưởng
- **K-Nearest Neighbors (KNN)**: thuật toán học máy cơ bản, thuộc nhóm Supervised.
- Được sử dụng cho bài toán phân loại và hồi quy.
- **Ý tưởng**: các điểm dữ liệu *gần nhau trong không gian đặc trưng* thường thuộc cùng một lớp hoặc giá trị tương tự.
- Nguyên tắc hoạt động dựa trên "láng giềng gần nhất", tức là dự đoán nhãn hoặc giá trị của một điểm dữ liệu mới dựa trên thông tin của các điểm dữ liệu gần nhất trong tập huấn luyện.

- KNN là thuật toán "lazy learner" hoặc "instance-based" $\to$ nó *không xây dựng mô hình cụ thể* trong quá trình huấn luyện; nó chỉ *lưu trữ toàn bộ dữ liệu tập huấn luyện* và chỉ thực hiện *tính toán khi có dữ liệu mới* cần dự đoán $\to$ tìm ra $K$ điểm dữ liệu gần với điểm dữ liệu mới để đưa ra dự đoán về nhãn hoặc giá trị.
## 2. Đặc điểm
- Đơn giản và dễ hiểu;
- Không tham số (Non-parametric) tức là không đòi hỏi bất kì giả định nào về phân phối dữ liệu;
- Lazy learning, không có giai đoạn huấn luyện rõ ràng nào, chỉ có lưu trữ và tính toán.
- Linh hoạt áp dụng cho cả phân loại và hồi quy;
- Phân loại đa lớp mà không cần điều chỉnh thuật toán;
- Cập nhập dữ liệu dễ dàng.
## 3. Ứng dụng
- Nhận diện mẫu (Pattern recognition): chữ viết tay, phân loại hình ảnh, ...
- Hệ thống đề xuất (Recommendation system): gợi ý phim, sản phẩm, ...
- Phát hiện bất thường (Anomaly detection)
- Chẩn đoán y tế, phân tích tài chính, xử lí ngôn ngữ tự nhiên, ...
# II. Toán học
## 1. Độ đo khoảng cách trong không gian đặc trưng
- Hiệu suất của KNN phụ thuộc lớn vào cách đo khoảng cách giữa các điểm dữ liệu.
- Một số độ đo khoảng cách phổ biến bao gồm:
### 1.1 Khoảng cách Euclidean
- Đây là độ đo phổ biến nhất, tính **khoảng cách đường thẳng** giữa hai điểm trong không gian Euclidean.
- Khoảng cách giữa hai điểm $x = (x_1, x_2, ..., x_n)$ và $y = (y_1, y_2, ..., y_n)$ được xác định:
  $$d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}$$
  phù hợp với dữ liệu liên tục và khi các đơn vị có cùng đơn vị đo hay phân phối chuẩn.
### 1.2 Khoảng cách Manhattan
- Còn được gọi là **khoảng cách L1**, tính tổng chênh lệch tuyệt đối theo từng chiều, ta có: $$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$phù hợp cho dữ liệu có nhiều giá trị bằng $0$ hoặc dữ liệu có nhiều giá trị rời rạc hoặc như không gian đô thị với đường di chuyển theo lưới.
### 1.3 Khoảng cách Minkowski
- Là **tổng quát hóa của cả khoảng Euclidean và Minkowski**, công thức:
  $$d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p \right)^{1/p}$$
  với:
	- $p = 1$: tương ứng với khoảng cách Manhattan;
	- $p = 2$: tương ứng với khoảng cách Euclidean;
	- Giá trị $p$ thường là một trong hai giá trị trên.
## 2. Lựa chọn tham số K trong KNN
- Việc chọn giá trị K (số lượng láng giềng gần nhất) là rất quan trọng và ảnh hưởng lớn đến hiệu suất của KNN:
	- Giá trị $K$ **nhỏ**:
	    - Dễ bị nhiễu, dẫn đến overfitting vì phụ thuộc quá nhiều vào một/một số ít điểm gần nhất đó.
	    - Biên quyết định phức tạp và không ổn định.
	    - Nhạy cảm với nhiễu và dữ liệu ngoại lai.
	- Giá trị $K$ **lớn**:
	    - Làm mềm ranh giới quyết định, có thể gây underfitting, bao gồm cả điểm không liên quan.
	    - Mất đi tính "local", giảm tính chất "láng giềng gần nhất".
- **Để chọn K tối ưu, thường sử dụng phương pháp cross-validation** để đánh giá hiệu suất của mô hình với các giá trị $K$ khác nhau và chọn giá trị $K$ cho kết quả phù hợp nhất.
- Giá trị $K$ thường được chọn là *số lẻ* để tránh trường hợp hòa (equal) trong bài toán phân loại nhị phân.
## 3. Curse of Dimensionality
- Trong không gian nhiều chiều, KNN có thể gặp phải vấn đề "lời nguyền chiều cao" (curse of dimensionality)
	- Khi số chiều của dữ liệu tăng lên, khoảng cách giữa các điểm trở nên ít ý nghĩa hơn và các điểm dữ liệu trở nên thưa thớt hơn.
	- Làm giảm hiệu suất của KNN vì các láng giềng gần nhất không còn thực sự "gần" trong không gian nhiều chiều.
- Để **cải thiện hiệu suất trong không gian nhiều chiều**, có thể sử dụng các *kỹ thuật giảm chiều* như PCA (Principal Component Analysis) hoặc *lựa chọn đặc trưng* (feature selection).
# III. Cài đặt KNN
## 1. From Scratch
- Quy trình bao gồm các bước:
	1. **Lưu trữ tập huấn luyện**: Lưu trữ toàn bộ dữ liệu huấn luyện ($X_\text{train}$, $y_\text{train}$).
	2. **Tính khoảng cách**: 
		- Với mỗi điểm dữ liệu mới cần dự đoán, tính khoảng cách đến tất cả các điểm trong tập huấn luyện.
		- Thường sử dụng khoảng cách Euclidean.
	3. **Tìm K láng giềng gần nhất**: Sắp xếp khoảng cách và chọn $K$ điểm gần nhất dựa trên khoảng cách nhỏ nhất.
	4. **Dự đoán**:
	    - **Phân loại**: Lấy lớp phổ biến nhất (majority vote) trong $K$ láng giềng.
	    - **Hồi quy**: Tính trung bình giá trị của $K$ láng giềng.
```python
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
	
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
	
    def _calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'minkowski':
            p = 3 # Mặc định p=3 cho Minkowski
            return np.power(np.sum(np.abs(x1 - x2) ** p), 1/p)
        else:
            raise ValueError("Không hỗ trợ độ đo khoảng cách này")
	
    def _predict_single(self, x):
        distances = [self._calculate_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
	
    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])
	
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
	
# TEST
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNNClassifier(k=3, distance_metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
print(f"Độ chính xác: {accuracy:.4f}")
```
## 2. Scikit-Learn
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# KNN Classifier --------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("KNN Classifier")
print("- Classification Report")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("- Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# KNN Regressor ----------------------------------------------
diabetes = load_diabetes() # Thay đổi ở đây
X, y = diabetes.data, diabetes.target # Thay đổi ở đây
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)
y_pred = knn_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n\nKNN Regressor")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")
```