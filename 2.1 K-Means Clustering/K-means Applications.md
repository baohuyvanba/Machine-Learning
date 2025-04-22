# I. Clustering:
- Ứng dụng K-means cho bài toán phân loại chữ số viết tay.
- Phân cụm chữ số viết tay là một bài toán kinh điển trong lĩnh vực học máy, thường được sử dụng làm *chuẩn mực để đánh giá* các thuật toán phân loại.
- Phương pháp tiếp cận không giám sát sử dụng K-means cung cấp những hiểu biết giá trị về cấu trúc nhóm tự nhiên trong dữ liệu chữ số mà không cần dựa vào các ví dụ đã được gán nhãn
## 1. MNIST
- [Bộ cơ sở dữ liệu MNIST](http://yann.lecun.com/exdb/mnist/) là bộ cơ sở dữ liệu lớn nhất về chữ số viết tay và được sử dụng trong hầu hết các thuật toán nhận dạng hình ảnh (Image Classification). MNIST bao gồm hai tập con:
	- Tập dữ liệu huấn luyện (training set) có tổng cộng 60k ví dụ khác nhau về chữ số viết tay từ 0 đến 9;
	- Tập dữ liệu kiểm tra (test set) có 10k ví dụ khác nhau.
## 2. Bài toán phân cụm
- Trong phân cụm chữ số viết tay với K-means, bài toán đặt ra là:
	- Cho một tập hợp các hình ảnh chữ số viết tay, mỗi hình ảnh được biểu diễn dưới dạng vector các giá trị pixel;
	- Xác định các nhóm tự nhiên sao cho các chữ số có đặc điểm thị giác tương đồng.
- Vì K-means không giám sát, không trực tiếp phân loại theo giá trị mà *nhóm các mẫu hình ảnh thị giác tương tự*.

- Việc đánh giá hiệu suất của K-means trong bài toán này có thể được thực hiện bằng cách so sánh kết quả phân cụm với nhãn gốc (ground truth labels) sử dụng các độ đo như homogeneity score, completeness score, V-measure, adjusted Rand index, adjusted mutual information, và silhouette coefficient.
- Các độ đo này cung cấp các góc nhìn khác nhau về chất lượng phân cụm, giúp đánh giá mức độ thuật toán nắm bắt cấu trúc tự nhiên của dữ liệu.
## 3. Cài đặt
```python
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import numpy as np

#Dataset digits
digits = datasets.load_digits()
X = digits.data
y = digits.target

#KMeans với K=10
K = 10
kmeans = KMeans(n_clusters = K, random_state = 0, init = 'k-means++')
kmeans.fit(X)
cluster_labels = kmeans.labels_

#Evaluation
ari = adjusted_rand_score(y, cluster_labels)
print("Adjusted Rand Index:", ari)

#Visualization
cluster_digits = [[] for _ in range(10)]
for digit, label in zip(y, cluster_labels):
    cluster_digits[label].append(digit)

fig, ax = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    ax[i//5, i%5].imshow(digits.data[cluster_digits[i][0]].reshape(8,8), cmap='gray')
    ax[i//5, i%5].set_title(f'Cluster {i}')

plt.tight_layout()
plt.show()
```
## 4. Đánh giá: chỉ số ARI
- Chỉ số **Adjusted Rand Index** (ARI) được sử dụng để đánh giá chất lượng phân cụm.
- Đây là một chỉ số *đo lường mức độ tương đồng giữa hai tập hợp phân cụm* (clustering). Nó được sử dụng để đánh giá độ chính xác của một thuật toán phân cụm so với một nhãn phân cụm thực tế (ground truth).
- **Công thức**:
  - ARI được tính dựa trên *Rand Index (RI)*, nhưng được điều chỉnh để loại bỏ sự ảnh hưởng của việc phân cụm ngẫu nhiên. Công thức của ARI là:
  $$\text{ARI} = \dfrac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}$$
    trong đó:
    - $\text{RI}$: Rand Index, tỉ lệ của các cặp điểm được phân vào cùng 1 cụm/khác cụm trong cả 2 phân cụm.
    - $E[\text{RI}]$: giá trị kỳ vọng của $\text{RI}$ khi phân cụm ngẫu nhiên.
    - Khoảng giá trị của $\text{ARI} \in [-1, 1]$, cụ thể:
      - $\text{ARI} = 1 \to$ 2 phân cụm giống nhau;
      - $\text{ARI} = 0 \to$ phân cụm gần như ngẫu nhiên;
      - $\text{ARI} = 1 \to$ phân cụm tệ hơn ngẫu nhiên (hiếm);
- **Ứng dụng**:
  - Đánh giá thuật toán phân cụm K-means, DBSCAN, ...
  - So sánh kết quả hoặc kiểm tra tính ổn định của phân cụm.
- **Đánh giá chung**:
  - Điều chỉnh tránh đánh giá sai do ngẫu nhiên, không bị ảnh hưởng bởi số cụm và hữu ích để so sánh.
  - Yêu cầu nhãn thực (không phù hợp bài toán phân cụm không giám sát thuần túy).
- **Ví dụ minh hoạ**:
	- Bảnh phân cụm:
    ![](.\data\Problem.png)
	- Tính giá trị **Rand Index - $\text{RI}$**: đo lường hai cách phân cụm. Ý tưởng chính là xét tất cả các cặp điểm dữ liệu trong tập dữ liệu, sau đó kiểm tra xem mỗi cặp có được *“agreement” hay “disagreement” giữa hai cách phân cụm*.
	- Cụ thể, với mỗi cặp ta xét hai vấn đề:
		1. Cùng/khác cụm trong cách phân loại (1) và phân loại (2);
		2. Cùng trạng thái cùng/khác trong hai cách phân loại không?
	- Một cặp dữ liệu có cùng trạng thái (cùng hoặc khác nhau) trong cả hai phân cụm, ta nói rằng cặp đó "agreement".
    $$\text{RI} = \dfrac{\text{số cặp agreement}}{\text{tổng số cặp}}  = \dfrac{a+b}{a+b+c+d}$$
    - Trong đó:
      - $a$ - số cặp mà cả hai cách **đều** cho vào trạng thái **cùng** cụm;
      - $b$ - số cặp mà cả hai cách **đều** cho vào trạng thái **khác** cụm;
  	  - $c$ - số cặp mà trạng thái **cùng** theo cách phân loại (1) nhưng **khác** theo cách phân loại (2);
	  - $d$ - số cặp mà trạng thái **khác** theo cách phân loại (1) nhưng **cùng** theo cách phân loại (2).
    - Khi đó ta được bảng sau:
    ![](.\data\RI.png)
	- Giá trị $\text{RI} = \dfrac{a+b}{a+b+c+d} = \dfrac{2+5}{15} \approx 0.4667$. Nghĩa là $46.67\%$ cặp được gán theo cùng 1 cách ở hai cách phân cụm.
	- Tuy nhiên, vấn đề của $\text{RI}$ là nó không hiệu chỉnh được việc các phân cụm ngẫu nhiên có thể tạo ra một giá trị không quá thấp. Vì vậy, ta sử dụng **Adjusted Rand Index (ARI)** – công thức này điều chỉnh bằng cách trừ đi giá trị kỳ vọng $E[\text{RI}]$ khi phân cụm ngẫu nhiên, rồi chia cho hiệu số giữa giá trị tối đa và $E[\text{RI}]$, ta được:
    $$\text{ARI} = \dfrac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}$$

- **Các bước tính toán** thực tế:
	- Từ bảng phân cụm, với danh sách $[A, B, C, D, E, F]$ ta có thể viết thành:
		- Ground truth: $[0, 0, 0, 1, 1, 1]$;
		- Predicted: $[0, 0, 1, 1, 1, 0]$.
	- Bảng Contingency (bảng giao nhau):
    ![](.\data\Contingency.png)
	  - Ví dụ như ô $(0,0)$ có hai điểm là $A, B \to n_{ij} = n_{00} = 2$ điểm; ... 
	- Tính giá trị $\displaystyle\sum_{ij}\begin{pmatrix} n_{ij} \\ 2 \end{pmatrix}$: tổng các cặp điểm có cùng cách phân loại ở 2 cách phân cụm
		- Vị trí $(0,0)$ có $n_{00} = 2$ điểm thì tạo thành 1 cặp, nên $\begin{pmatrix} n_{00} \\ 2 \end{pmatrix} = \begin{pmatrix} 2 \\ 2 \end{pmatrix} = 1$.
		- Vị trí $(0,1)$ có $n_{01} = 1$ điểm thì không thể tạo cặp, nên $\begin{pmatrix} n_{01} \\ 2 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix} = 0$.
		- Tương tự, vị trí $(1,0)$ được $\begin{pmatrix} 1 \\ 2 \end{pmatrix} = 0$ và $(1,1)$ được $\begin{pmatrix} 2 \\ 2 \end{pmatrix} = 1$.
		- Khi đó: tổng có giá trị $1+0+0+1 = 2$.
	- Tính giá trị $\displaystyle\sum_{i}\begin{pmatrix} a_{i} \\ 2 \end{pmatrix}$: tổng các cặp điểm có các phân loại vào cùng 1 cụm, theo nhãn thực tế - hàng
		- Vị trí hàng $0$ có $3$ điểm, nên $\begin{pmatrix} 3 \\ 2 \end{pmatrix} = 3$
		- Vị trí hàng $1$ có $3$ điểm, nên $\begin{pmatrix} 3 \\ 2 \end{pmatrix} = 3$
		- Ta được tổng $3+3=6$.
	- Tính giá trị $\displaystyle\sum_{j}\begin{pmatrix} b_{j} \\ 2 \end{pmatrix}$: tổng các cặp điểm có các phân loại vào cùng 1 cụm, theo nhãn dự đoạn - cột
		- Vị trí cột $0$ có $3$ điểm, nên $\begin{pmatrix} 3 \\ 2 \end{pmatrix} = 3$
		- Vị trí cột $1$ có $3$ điểm, nên $\begin{pmatrix} 3 \\ 2 \end{pmatrix} = 3$
		- Ta được tổng $3+3=6$.
	- Tổng các nhóm được tạo ra $\begin{pmatrix} 6 \\ 2 \end{pmatrix} = 15$ cặp.
	- *Công thức ARI*:
	$$\text{ARI} = \dfrac{\sum_{ij}\begin{pmatrix}n_{ij}\\2\end{pmatrix} - \dfrac{\sum_i\begin{pmatrix}a_{i}\\2\end{pmatrix} \times \sum_j\begin{pmatrix}b_{j}\\2\end{pmatrix}}{\begin{pmatrix}n\\2\end{pmatrix}}}{\dfrac{1}{2}\left( \sum_i\begin{pmatrix}a_{i}\\2\end{pmatrix} + \sum_j\begin{pmatrix}b_{j}\\2\end{pmatrix} \right) - \dfrac{\sum_i\begin{pmatrix}a_{i}\\2\end{pmatrix} \times \sum_j\begin{pmatrix}b_{j}\\2\end{pmatrix}}{\begin{pmatrix}n\\2\end{pmatrix}}}$$
	- Khi đó ta tính được: $\text{ARI} = \dfrac{2 - \frac{6\times 6}{15}}{\frac{1}{2}(6+6) - \frac{6\times 6}{15}} \approx -0.1111$.

# II. Object Segmentation
## 1. Vấn đề
- Phân vùng đối tượng trong ảnh (object segmentation) là quá trình *phân chia một ảnh thành nhiều vùng* để đơn giản hóa biểu diễn hoặc phân tích.
- Mục tiêu là *xác định và phân tách các đối tượng hoặc vùng khác nhau* trong ảnh.
- Ứng dụng của phân vùng đối tượng rất rộng rãi, từ y tế (phân tích ảnh y tế) đến xe tự lái (nhận biết đường và đối tượng xung quanh).

- Thách thức chính là *xác định pixel nào thuộc cùng một đối tượng/vùng* chỉ dựa trên đặc điểm thị giác của chúng mà không cần thông tin ngữ nghĩa trước.
## 2. Ý tưởng
- K-means cung cấp một phương pháp tiếp cận trực quan cho bài toán phân vùng đối tượng bằng cách *nhóm các pixel tương tự nhau về màu sắc*.
- **Ý tưởng cơ bản**: là coi mỗi pixel trong ảnh là một điểm dữ liệu trong không gian màu (ví dụ: không gian màu RGB) và sử dụng K-means để phân cụm các pixel này.
- Các pixel thuộc cùng một cụm được giả định là có màu sắc tương tự và có thể thuộc cùng một đối tượng hoặc vùng.

- Số lượng cụm K trong K-means sẽ quyết định số lượng vùng phân đoạn trong ảnh kết quả.
- Việc lựa chọn K là rất quan trọng:
	- Quá ít cụm có thể làm gộp các đối tượng khác nhau;
	- Quá nhiều cụm có thể chia nhỏ một đối tượng thành nhiều phần.
## 3. Cài đặt
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Read image
image = cv2.imread('./data/lily.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #RGB

#Preprocess: Image -> 2D array of pixel, float32
pixel_vals = image.reshape((-1, 3))
pixel_vals = np.float32(pixel_vals)

#n-clusters
k = 5

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

#K-means
ret, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#Convert centers to uint8
centers = np.uint8(centers)

#Assign each pixel to its cluster center
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

#VisualizeVisualize
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Original")
plt.imshow(image)
plt.subplot(122)
plt.title("Segmented Image")
plt.imshow(segmented_image)
plt.show()
```
- Ảnh đầu vào được chuyển đổi thành không gian màu $\text{RGB}$ và sau đó chuyển thành mảng 2 chiều `numpy` với 3 cột ứng với 3 giá trị $\text{R, G, B}$ và mỗi dòng tương ứng với mỗi pixel.
- Xác định tiêu chí dừng thuật toán:
	- `cv2.TERM_CRITERIA_EPS`: sự thay đổi tâm cụm nhỏ hơn một giá trị Epsilon ($0.85$);
	- `cv2.TERM_CRITERIA_MAX_ITER`: số lần lặp tối đa ($100$).
- Thuật toán: `ret, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)`
	- `bestLabels = None` - cung cấp nhãn tốt nhất từ lần chạy K-means trước đó (chúng ta không có, nên để None);
	- `criteria` - Tiêu chí dừng thuật toán;
	- `attempts = 10` - số lần thuật toán K-means sẽ được chạy với các tâm cluster ban đầu khác nhau.
		- OpenCV sẽ chạy K-means 10 lần và trả về kết quả tốt nhất (với độ nén cụm tốt nhất).
		- Giúp giảm khả năng thuật toán bị mắc kẹt vào một local minimum (điểm tối ưu cục bộ) không phải global optimum (điểm tối ưu toàn cục).
	- `cv2.KMEANS_RANDOM_CENTERS` - phương pháp khởi tạo tâm cluster ban đầu, chọn ngẫu nhiên.
	- **Giá trị trả về**:
		- **`ret`**: giá trị độ nén (compactness).
			- Tổng bình phương khoảng cách từ mỗi điểm đến tâm cluster gần nhất của nó.
			- Có thể được dùng để đánh giá chất lượng phân cụm.
		- **`labels`**: mảng 1 chiều chứa nhãn cluster cho mỗi pixel.
			- Mỗi phần tử trong mảng này là một số nguyên từ 0 đến `k-1`, chỉ ra pixel đó thuộc về cluster nào.
			- Mảng này có cùng số lượng phần tử với số lượng pixel trong ảnh ban đầu.
		- **`centers`**: mảng 2 chiều chứa tọa độ tâm của `k` cluster.
			- Mỗi hàng trong mảng này là tâm của một cluster, và mỗi tâm cluster có 3 giá trị RGB (vì chúng ta đang phân cụm dựa trên màu RGB).
# III. Image Compression
## 1. Nguyên lý Lượng tử hóa Vector (Vector Quantization)
- Nén ảnh bằng K-means hoạt động dựa trên *nguyên lý lượng tử hóa vector để giảm số lượng màu sắc khác biệt* được sử dụng để biểu diễn ảnh.
- Trong không gian RGB tiêu chuẩn, mỗi pixel cần $8\times 3 = 24$ bits thông tin (16 triệu màu).
- Tuy nhiên, hầu hết hình ảnh chỉ sử dụng một phần nhỏ của không gian màu này.

- K-means: *nhóm các màu sắc pixel tương tự vào $K$ cụm*, với số lượng nhỏ hơn đáng kể không gian màu gốc.
- Mỗi centroid của mỗi cụm trở thành màu đại diện, quá trình nén xảy ra bằng việc *chỉ lưu trữ chỉ số cụm cho mỗi pixel* thay vì toàn bộ giá trị 3 kênh màu.
- Do đó, tỉ lệ nén sẽ tăng lên khi $K$ giảm.
- Đây là *kĩ thuật nén mất mát dữ liệu* (lossy compression) vì ảnh được nén gần tương tự với ảnh gốc nhưng sẽ có đánh đối giữa chất lượng ảnh nén với tỉ lệ nén bằng việc căn chỉnh giá trị $K$.
## 2. Toán học
- **Mục tiêu**: tìm $K$ màu đại diện để giảm thiểu tổng lỗi lượng tử hóa, được biểu diễn bằng tổng bình phương sai số trung bình giữa giá trị pixel gốc và centroid tương ứng của chúng.
- Hàm mục tiêu:
  $$J = \dfrac{1}{N} \displaystyle\sum{\lVert x^{(n)} - \mu_{z^{(n)}} \lVert^2}$$
  trong đó:
	- $x^{(n)}$: giá trị pixel gốc thứ $n$.
	- $z^{(n)}$: chỉ số centroid được gán cho pixel thứ $n$.
	- $\mu_{z^{(n)}}$: giá trị centroid của cụm $z^{(n)}$.
	- $N$: tổng số pixel trong ảnh.
- Quá trình mã hóa ánh xạ mỗi pixel đến chỉ số cụm của nó, do đó chỉ cần $\log_2(K)$ bit để biểu diễn.
- Quá trình giãi mã thực hiện việc ánh xạ ngược lại.
- **Tỷ lệ nén** có thể được tính:
  $$\dfrac{24}{\log_2(K) + \text{overhead}}$$
  với $\text{overhead}$ là chi phí lưu trữ các centroids.
## 3. Quy trình nén thực tế
1. **Tiền xử lý ảnh**:
	- Chuyển đổi ảnh sang không gian màu phù hợp (thường là RGB).
	- Chuyển đổi ảnh thành mảng 2D, mỗi hàng đại diện cho một pixel.
2. **Phân cụm K-means**:
	- Áp dụng thuật toán K-means để phân cụm các giá trị pixel, với số lượng cụm $K$ quyết định mức độ nén.
	- $K$ càng nhỏ, tỷ lệ nén càng cao nhưng chất lượng ảnh có thể giảm.
3. **Tạo biểu diễn nén:** Lưu trữ
	- $K$ centroids (mỗi centroid là giá trị RGB 24-bit);
	- Chỉ số cụm cho mỗi pixel ($\log_2(K)$ bit).
4. **Giải nén**:
	- Tái tạo ảnh bằng cách ánh xạ mỗi chỉ số cụm trở lại màu centroid tương ứng.
## 4. Cài đặt
```python
#Init centroids randomly
def initialize_means(points, clusters):
    indices = np.random.choice(len(points), clusters, replace=False)
    means   = points[indices, :]
    return means
```
- Hàm khởi tạo ngẫu nhiên Clusters:
	- Đầu vào là tập các điểm (mảng 2 chiều) và số lượng clusters mong muốn;
	- Trả về một mảng 2 chiều các điểm centroids được khởi tạo.

```python
#Euclidean distance
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

#K-means
def k_means(points, means, clusters, max_iters=10):
    for _ in range(max_iters):
        indices = np.array([np.argmin([distance(point, mean) for mean in means]) for point in points])
        for i in range(clusters):
            points_in_cluster = points[indices == i]
            if len(points_in_cluster) > 0:
                means[i] = np.mean(points_in_cluster, axis=0)
    return means, indices
```
- Hàm K-means được xây dựng thủ công:
	- Nhận vào các tham số:
		- `points`: Dữ liệu điểm;
		- `means`: Tâm cluster ban đầu;
		- `clusters`: Số lượng cluster;
		- `max_iters`: Số lần lặp tối đa của thuật toán K-means (mặc định là 10).
	- Vòng lặp:
		- Gán các điểm vào Clusters gần nhất (trả về một mảng chỉ số cluster cho các điểm đầu vào);
		- Tính toán lại giá trị centroid mới.
	- Trả về: giá trị các centroids và mảng chỉ số clusters.

```python
#Compression function
def compress_image(means, index, img, clusters):
    compressed_image = means[index].reshape(img.shape)
    cv2.imwrite(f'compressed_{clusters}_colors.png', (compressed_image * 255).astype(np.uint8))
    plt.figure(figsize=(8, 6))
    plt.title(f"Ảnh nén ({clusters} màu)")
    plt.imshow(compressed_image)
    plt.axis('off')
    plt.show()

# n_clusters = n_colors
clusters = 16
means = initialize_means(points, clusters)
means, index = k_means(points, means, clusters)
compress_image(means, index, image, clusters)
```
