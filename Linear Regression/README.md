# I. Introduction
> [!NOTE]
> Một chiếc ô tô có động cơ dung tích $x_1$ lít, số ghế $x_2$ và đã đi được $x_3$ km thì có giá bao nhiêu?

- Giả sử có thống kê từ **1000** chiếc ô tô đã bán trên thị trường, liệu rằng với các thông số trên *ta có thể dự đoán giá* của chiếc ô tô này không?
- Ta xây dựng một **Hàm dự đoán**: $y = f(x)$ với $x = [x_1, x_2, x_3]$ là vector chứa thông tin input và $y$ là thông tin output.
- Một số mối quan hệ đơn giản có thể nhận thấy:
  1. Dung tích động cơ càng lớn thì giá ô tô thường cao hơn;
  2. Số ghế càng nhiều thì giá ô tô có xu hướng cao hơn;
  3. Số km đã đi càng nhiều thì giá ô tô sẽ giảm.
- Một mô hình đơn giản có thể mô tả mối quan hệ giữa giá ô tô và các thông số đầu vào là:
  $$
  \begin{matrix} 
    y \approx f(x) = \hat{y} \\ 
    f(x) = w_1x_1 + w_2x_2 + w_3x_3 + w_0 
  \end{matrix}
  $$
  trong đó: $w_1, w_2, w_3$ là các trọng số và $w_0$ là giá trị bias.
- Mối quan hệ $y \approx f(x)$ bên trên là một mối quan hệ tuyến tính (linear).
- Bài toán này là một bài toán thuộc loại Regression (Hồi quy): đi tìm các hệ số tối ưu $\{w_1, w_2, w_3, w_0\}$ chính vì vậy được gọi là bài toán **Linear Regression**.
- **Chú ý**:
	- $y$ là giá trị thực (dựa trên số liệu thống kê chúng ta có trong tập *training data*), và $\hat{y}$ là giá trị mà mô hình đưa ra.
	- Nhìn chung, $y$ và $\hat{y}$ là hai giá trị khác nhau do có sai số mô hình $\to$ ta mong muốn rằng sự khác nhau này rất nhỏ.
	- *Linear* hay *tuyến tính* hiểu một cách đơn giản là thẳng, phẳng.
		- Không gian 2 chiều: hàm số *tuyến tính* - đồ thị dạng *đường thẳng*.
		- Không gian 3 chiều: hàm số *tuyến tính* - đồ thị dạng *mặt phẳng*.
		- Không gian nhiều hơn 3 chiều: *siêu phẳng (hyperplane)*.
		- Các hàm số tuyến tính là các hàm đơn giản nhất, vì chúng thuận tiện trong việc hình dung và tính toán.
# II. Toán học
## 1. Linear Regression
- Đặt:
	- $w=[w_0, w_1, w_2, w_3]^T$ là vector (cột) hệ số cần phải tối ưu;
	- $\bar{x} = [1, x_1, x_2, x_3]$ là vector (hàng) dữ liệu đầu vào.
    	- Giá trị $1$ (bias) ở đầu được thêm vào để phép tính đơn giản hơn và thuận tiện cho việc tính toán.
- Khi đó ta được phương trình:
  $$ y \approx \hat{y} = \bar{x}w $$
## 2. Sai số dự đoán
- Ta mong muốn rằng *sự sai khác* $e$ (error) giữa giá trị thực $y$ và giá trị dự đoán $\hat{y}$ đạt nhỏ nhất.
- Tương ứng:
  $$\dfrac{1}{2}e^2 = \dfrac{1}{2}(y-\hat{y})^2= \dfrac{1}{2}(y-\bar{x}w)^2$$
  hệ số $\dfrac{1}{2}$ để triệt tiêu trong quá trình đạo hàm.
- Ta cần giá trị $e^2$ đạt nhỏ nhất, thay vì $e$ nhỏ nhất do giá trị $e$ có thể âm.
## 3. Hàm mất mát
- Điều tương tự xảy ra với tất cả các cặp (input, output) $(\text{x}_i, y_i), i = 1, 2, \dots, N$ với $N$ là số lượng dữ liệu quan sát được.
- Mong muốn: *tổng sai số là nhỏ nhất*, tương đương với việc tìm $w$ để hàm số sau đạt GTNN:
  $$\mathcal{L}(\text{w}) = \dfrac{1}{2}\sum_{i=1}^{N}{(y_i - \bar{\text{x}}_i\text{w})^2}$$
- Hàm số $\mathcal{L}(w)$ ở trên gọi là **hàm mất mát** của bài toán **Linear Regression**, yêu cầu khi đó của ta là sai số này có giá trị nhỏ nhất $\to$ tìm vector hệ số $w$: gọi là *điểm tối ưu* (optimal point)
  $$\text{w}^* = \arg\min_{\text{w}}{\mathcal{L}(\text{w})}$$
- Trước khi đi đến lời giải, ta đơn giản hóa hàm mất mát ở trên với:
	- $\bar{\text{X}} = [\bar{\text{x}}_1, \bar{\text{x}}_2, \dots, \bar{\text{x}}_N]$: ma trận đầu vào, mỗi dòng là một điểm dữ liệu;
	- $\text{y} = [y_1, y_2, \dots, y_N]$: vector cột chứa output;
- Ta được:
  $$\mathcal{L}{(\text{w})} = \dfrac{1}{2}\sum_{i=1}^{N}{(y_i - \bar{\text{x}}_i\text{w})^2} = \dfrac{1}{2} \lVert \text{y} - \bar{\text{X}}\text{w} \lVert_2^2$$
  với $\lVert \text{z} \lVert_2$ là **Euclidean Norm** (chuẩn Euclid - khoảng cách Euclid), nói cách khác $\lVert z \lVert_2^2$ là *tổng bình phương* mỗi phần tử trong vector $\text{z}$.
## 4. Nghiệm cho bài toán Linear Regression
- Cách tiếp cận đơn giản từ trước là giải phương trình đạo hàm (gradient) bằng 0 - không quá phức tạp và với phương trình tuyến tính thì khả thi.
- Đạo hàm theo $\text{w}$ của hàm mất mát:
  $$\dfrac{\partial \mathcal{L}(\text{w})}{\partial \text{w}} = \bar{\text{X}}^T (\bar{\text{X}}\text{w} - y)$$
- Đạo hàm vector: [Source](https://ccrma.stanford.edu/~dattorro/matrixcalc.pdf)
- Phương trình đạo hàm trên tương đương với:
  $$\bar{\text{X}}^T \bar{\text{X}}\text{w} = \bar{\text{X}}^Ty \triangleq \text{b}$$
  với $\bar{\text{X}}^Ty \triangleq \text{b}$, tức là đặt $\text{b} = \bar{\text{X}}^Ty$.
- Nếu ma trận $\text{A} \triangleq \bar{\text{X}}^T \bar{\text{X}}$ khả nghịch (non-singular hay invertible) thì phương trình trên có nghiệm duy nhất:
  $$\text{w} = \text{A}^{-1}\text{b}$$
  ngược lại, $\text{A}$ không khả nghịch (Det bằng 0) thì phương trình vô nghiệm/vô số nghiệm.
- Ta sử dụng khái niệm giả nghịch đảo $\text{A}^\dagger$ (A dagger), khi đó, *điểm tối ưu cho bài toán Linear Regression*:
  $$\text{w} = \text{A}^\dagger\text{b} = (\bar{\text{X}}^T\bar{\text{X}})^\dagger \bar{\text{X}}^T\text{y}$$

# III. Ứng dụng trên Python

Ví dụ ta có bài toán:
| Chiều cao (cm) | Cân nặng (kg) | Chiều cao (cm) | Cân nặng (kg) |
| -------------- | ------------- | -------------- | ------------- |
| 147            | 49            | 168            | 60            |
| 150            | 50            | 170            | 72            |
| 153            | 51            | 173            | 63            |
| 155            | 52            | 175            | 64            |
| 158            | 54            | 178            | 66            |
| 160            | 56            | 180            | 67            |
| 163            | 58            | 183            | 68            |
| 165            | 59            |                |               |
## 1. Dựa theo công thức
```python
# Building Xbar
one  = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
```
- $\text{A} \triangleq \bar{\text{X}}^T \bar{\text{X}}$ : `np.dot(Xbar.T, Xbar)`
- $\text{b} \triangleq \bar{\text{X}}^Ty$ : `np.dot(Xbar.T, y`
- $\text{w} = \text{A}^\dagger \text{b}$ : `np.dot(np.linalg.pinv(A), b)`

- Vẽ fitting-line:
```python
# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0

# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')       # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
```
## 2. Sử dụng thư viện Scikit-learn
```python
from sklearn import datasets, linear_model

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by (5): ', w.T)
```

# IV. Thảo luận
## 1. Các bài toán có thể áp dụng Linear Regression
- Hàm số $y \approx f(\text{x}) = \text{w}^T\text{x}$ là một hàm tuyến tính theo cả $\text{w}$ và $\text{x}$.
- Trên thực tế, Linear Regression có thể áp dụng cho các mô hình chỉ cần tuyến tính theo $\text{w}$.
## 2. Hạn chế của Linear Regression
- Nhạy cảm với nhiễu - cần phải có bước tiền xử lí để loại bỏ ảnh hưởng của nhiễu.
- Không biểu diễn được các mô hình phức tạp.

# Tham khảo
- https://machinelearningcoban.com/2016/12/28/linearregression/