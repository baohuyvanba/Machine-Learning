# Machine Learning Algorithm Implementations
This repository contains implementations of various machine learning algorithms, covering both library-based and from-scratch approaches.

## Algorithms Implemented
- [x] Linear Regression
- [ ] Logistic Regression
- [ ] Decision Trees
- [ ] Random Forests
- [ ] Support Vector Machines (SVM)
- [ ] K-Nearest Neighbors (KNN)
- [x] K-Means Clustering
- [x] DBSCAN
- [ ] Principal Component Analysis (PCA)
- [ ] Neural Networks

****

# Machine Learning Fundamentals
- Là một nhánh của trí tuệ nhân tạo, máy tính học từ dữ liệu $\to$ quyết định, không cần lập trình.
- Tập trung vào việc tìm kiếm mẫu trong dữ liệu $\to$ dự đoán hoặc phân loại dữ liệu mới.
## Các thuật toán Machine Learning cơ bản
- Học có giám sát (Supervised Learning)
  - Hồi quy tuyến tính, Cây quyết định, SVM, KNN.
- Học không giám sát (Unsupervised Learning)
  - K-Means, PCA, Gaussian Mixture.
- Học tăng cường (Reinforcement Learning)
  - Q-Learning, Deep Q-Networks (DQN).
## Các bước cơ bản trong Machine Learning
1. *Thu thập dữ liệu* : chất lượng và đủ lớn.
2. *Tiền xử lý dữ liệu* : Làm sạch, chuẩn hóa, chọn đặc trưng.
3. *Chọn thuật toán*
4. *Huấn luyện mô hình*
5. *Đánh giá mô hình* : Kiểm tra hiệu suất mô hình - Accuracy, Precision, Recall, F1-Score.
6. *Triển khai mô hình* : Đưa mô hình vào sử dụng thực tế và tiếp tục cải thiện hiệu suất.
## Phân loại thuật toán Machine Learning
### Theo phương thức học
1. **Học Có Giám Sát (Supervised Learning)**:
   - Dữ liệu gán nhãn $\to$ dự đoán đầu ra cho dữ liệu mới.
   - *Classification* (Phân loại): Nhãn thuộc một số **nhóm hữu hạn**. Ví dụ: phân loại thư rác.
   - *Regression* (Hồi quy): Nhãn là các **giá trị thực**. Ví dụ: dự đoán giá nhà.
2. **Học Không Giám Sát (Unsupervised Learning)**:
   - Học từ dữ liệu chưa được gán nhãn, chỉ có dữ liệu đầu vào.
   - Tìm cấu trúc ẩn trong dữ liệu, ví dụ: phân nhóm (clustering), giảm chiều dữ liệu.
   - *Clustering* (Phân nhóm): nhóm dựa trên sự **liên quan** giữa các dữ liệu.
   - *Association* (Kết hợp): Tìm **quy luật** dựa trên nhiều dữ liệu. Ví dụ: gợi ý sản phẩm mua kèm.
3. **Học Bán Giám Sát (Semi-supervised)**:
   - Học có giám sát + Học không giám sát.
   - Một phần nhỏ có nhãn.
4. **Học Tăng Cường (Reinforcement Learning)**:
   - Tự động **xác định hành vi** để đạt lợi ích cao nhất trong một môi trường.
   - Dựa trên **Lý Thuyết Trò Chơi**, tìm nước đi tiếp theo để đạt điểm cao nhất.
### Theo chức năng
- **Regression (Thuật Toán Hồi Quy)**:
  - *Linear Regression* (Hồi quy tuyến tính): Dự đoán giá trị liên tục - quan hệ tuyến tính.
  - *Logistic Regression* (Hồi quy Logistic): Nhóm (nhị phân) dựa trên xác suất.
  - *Stepwise Regression*: Lựa chọn biến độc lập từng bước để xây dựng mô hình hồi quy tốt hơn.
- **Classification (Thuật Toán Phân Loại)**:
  - *Linear Classifier* (Phân loại tuyến tính): Dựa trên hàm phân tách tuyến tính.
  - *Support Vector Machine* (SVM): Tìm đường ranh giới tối ưu phân tách các lớp dữ liệu.
  - *Kernel SVM*: (Không giải thích thêm trong đoạn văn).
  - *Sparse Representation-based classificatio*n (SRC): Phân loại dựa trên biểu diễn thưa thớt mẫu mới so với mẫu huấn luyện.
- **Instance-based (Thuật Toán Dựa Vào Biểu Diễn Mẫu)**:
  - *k-Nearest Neighbor* (kNN): Phân loại dựa trên k láng giềng gần nhất.
  - *Learning Vector Quantization* (LVQ): Biến thể của kNN, dùng vector mẫu học được.
- **Regularization (Thuật Toán Chính Quy)**:
  - *Ridge Regression* : Giảm overfitting bằng cách xử lý hệ số hồi quy lớn.
  - *Least Absolute Shrinkage and Selection Operator (LASSO)*: Giảm overfitting bằng cách đưa một số hệ số hồi quy về 0, chọn biến quan trọng.
  - *Least-Angle Regression (LARS)*: Tương tự LASSO, chọn biến theo hướng giảm lỗi từng bước.
- **Bayesian (Thuật Toán Bayes)**:
  - *Naive Bayes* : Dựa trên định lý Bayes để tính xác suất mẫu thuộc lớp nhất định.
  - *Gaussian Naive Bayes* : Giả sử thuộc tính độc lập và phân bố Gauss.
- **Clustering (Thuật Toán Phân Nhóm)**:
  - *k-Means clustering* : Phân nhóm dựa trên khoảng cách Euclid đến tâm cụm.
  - *k-Medians* : Tương tự k-Means, dùng khoảng cách Manhattan.
  - *Expectation Maximization (EM)*: Phù hợp dữ liệu có thành phần ẩn.
  - **Artificial Neural Network (Thuật Toán Mạng Nơ-ron Nhân Tạo)**:
  - *Perceptron* : Đơn vị cơ bản của mạng nơ-ron.
  - *Softmax Regression* : Phân loại đa lớp, đưa ra xác suất cho từng lớp.
  - *Multi-layer Perceptron* : Mạng nhiều lớp ẩn, học hàm phi tuyến phức tạp.
  - *Back-Propagation* : Thuật toán huấn luyện mạng nơ-ron bằng cách truyền ngược lỗi.
- **Dimensionality Reduction (Thuật Toán Giảm Chiều Dữ Liệu)**:
  - *Principal Component Analysis (PCA)* : Phân tích thành phần chính.
  - *Linear Discriminant Analysis (LDA)* : Phân tích biệt lượng tuyến tính.
- **Ensemble (Thuật Toán Tập Hợp)**:
  - *Boosting* : Tạo nhiều mô hình yếu và kết hợp thành mô hình mạnh.
  - *AdaBoost* : Thuật toán boosting phổ biến, tập trung vào mẫu dữ liệu sai.
  - *Random Forest* : Tạo nhiều cây quyết định và lấy trung bình/biểu quyết.