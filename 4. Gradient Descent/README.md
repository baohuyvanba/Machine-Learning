# I. Introduction
## 1. Ví dụ
- Xét hàm số như hình:
  ![](.\attachments\Example.png)
- Ở hàm số trên, điểm $x^*$ là điểm cực tiểu (local minimum) - cũng là điểm làm cho hàm số đạt GTNN.
	- Local minimum: cực tiểu cục bộ.
	- Global minimun: cực tiểu toàn cục (làm cho hàm số đạt GTNN), trường hợp đặc biệt của local minimum.
- Trong hình trên, càng xa về trái của điểm local minimum thì đạo hàm càng âm, càng xa về phải thì đạo hàm càng dương.

- Một hàm số 1 biến có đạo hàm trong miền xác định:
	1. Điểm local minimum $x^∗$ của hàm số là điểm có đạo hàm $f^′(x^∗) = 0$.
	2. Trong lân cận:
		- Đạo hàm phía bên trái $x^∗$ là không dương ($\le 0$);
		- Đạo hàm phía bên phải $x^∗$ là không âm ($\ge 0$).
	3. Đường tiếp tuyến với đồ thị hàm số đó tại 1 điểm bất kỳ có hệ số góc bằng đạo hàm của hàm số tại điểm đó.

## 2. Gradient Descent
- Trong Machine Learning nói riêng và Toán Tối Ưu nói chung, chúng ta thường xuyên phải tìm giá trị nhỏ nhất (hoặc đôi khi là lớn nhất) của một hàm số nào đó.
- Ví dụ như trong việc sử dụng hàm mất mát ở các mô hình Linear Regression, K-mean Clustering.
- Nhìn chung, *việc tìm global minimum của các hàm mất mát trong Machine Learning rất phức tạp* (bất khả thi):
	- Đạo hàm bằng $0$ $\to$ các điểm cực tiểu $\to$ tìm điểm GTNN $\to$ *bất khả thi* do dữ liệu có số chiều/số điểm dữ liệu lớn.
	- Cố gắng tìm các điểm local minium và coi đó là nghiệm cần tìm (mức độ nào đó).
- **Hướng tiếp cận phổ biến**: từ một điểm mà ta coi là *gần* với nghiệm $\to$ lặp để *tiếp dần* về điểm cần tìm. Trong đó, Gradient Descent là một biến thể thường được sử dụng.

- **Gradient Descent**: thuật toán tối ưu hóa bậc nhất được sử dụng rộng rãi để tìm giá trị nhỏ nhất cục bộ của một hàm số.
- Thuật toán quan trọng trong Machine Learning, được dùng để huấn luyện mô hình bằng cách giảm thiểu hàm mất mát.
- Nó hướng đến việc tìm kiếm các điểm cực tiểu cục bộ (local minimum), và trong nhiều trường hợp, các điểm cực tiểu cục bộ này đủ tốt để coi là nghiệm chấp nhận được cho bài toán
- Nguyên tắc:
	- Di chuyển từng bước nhỏ theo hướng ngược lại với gradient của hàm số tại điểm hiện tại.
	- Gradient, hay đạo hàm, cho biết hướng tăng nhanh nhất của hàm số - đi ngược hướng gradient là cách tiếp cận để tìm đến vùng có giá trị hàm số nhỏ hơn, tức là tìm đến điểm cực tiểu.
# II. Gradient Descent hàm 1 biến
## 1. Lý thuyết
- Xét hàm số $f(x)$ có đạo hàm trong miền xác định, xác định giá trị $x^*$ để $f(x^*)$ đạt cực tiểu.
- **Gradient Descent** tiếp cận bằng các vòng lặp:
	- Giả sử điểm $x_t$ là điểm tìm được sau vòng lặp thứ $t$. Khi đó, ta cần một thuật toán để đưa $x_t$ về gần với $x^*$.
	- Ta có các quan sát:
		- Nếu $f'(x_t) > 0 \longrightarrow x^t$ bên phải $x^*$. Để tìm $x_{t+1}$ thì ta cần di chuyển về phía bên trái (hướng âm), tức là đi *ngược dấu với giá trị đạo hàm* : $$x_{t+1} = x_t + \Delta$$với $\Delta$ là đại lượng *ngược dấu* với giá trị đạo hàm $f'(x_t)$.
		- $x_t$ càng xa $x^*$ về phía bên phải, giá trị $f'(x_t)$ càng lớn
			- Lượng di chuyển $\Delta \sim 1/f'(x_t)$ Do nó ngược dấu.
			- Hay chuyển thành: $\Delta \sim -f'(x_t)$.
	- Từ đó, ta có một *cách cập nhập* đơn giản: $$x_{t+1} = x_t - \eta f'(x_t)$$trong đó:
		- $\eta$ (eta) là tốc độ học - learning rate, quyết định kích thước bước nhảy - ảnh hưởng tốc độ hội tụ.
		- Dấu "$-$" thể hiện việc đi ngược đạo hàm.
	- Về mặt hình học, tại mỗi bước lặp, thuật toán Gradient Descent "lăn" từ từ xuống "đáy" của đồ thị hàm số. Đường tiếp tuyến tại một điểm trên đồ thị hàm số có hệ số góc chính bằng đạo hàm của hàm số tại điểm đó.
	- Gradient Descent chỉ đảm bảo tìm được đểm cực tiểu cục bộ, không nhất thiết là cực tiểu toàn cục. Điều này có nghĩa là *thuật toán có thể bị "mắc kẹt"* tại một "thung lũng" nhỏ trên bề mặt hàm số
## 2. Learning rate (tốc độ học)
- Tốc độ hội tụ Gradient Descent không những phụ thuộc vào điểm khởi tạo ban đầu mà còn bị ảnh hưởng bởi tốc độ học.
- Xét với bài toán ở trên với cùng giá trị khởi tạo ban đầu $x=-5$.
	1. Với  $\eta = 0.01$, tốc độ hội tụ rất chậm.
		Trong thực tế, khi việc tính toán trở nên phức tạp, _learning rate_ quá thấp sẽ ảnh hưởng tới tốc độ của thuật toán rất nhiều, thậm chí không bao giờ tới được đích.
	2. Với $\eta = 0.5$, thuật toán tiến rất nhanh tới gần đích sau vài vòng lặp.
		Tuy nhiên, thuật toán không hội tụ được vì _bước nhảy_ quá lớn, khiến nó cứ quẩn quanh ở đích.
## 3. Ví dụ
- Xét hàm số $f(x)=x^2+3e^{−x}$ với đạo hàm xác định $f'(x) = 2x-3e^{-x}$.
- Hàm số này kết hợp giữa đa thức và hàm mũ, khiến việc giải phương trình đạo hàm bằng 0 theo cách giải tích trở nên khó khăn. Do đó, Gradient Descent là một công cụ hữu ích để tìm điểm cực tiểu cục bộ của hàm số này.
- **Cài đặt thuật toán**:
```python
import math
import numpy as np
import matplotlib.pyplot as plt

def cost(x):
    #Tính giá trị hàm số f(x)= x^2 + 3e^(-x)
    return x**2 + 3*np.exp(-x)
    
def grad(x):
    #Tính giá trị đạo hàm của f(x)= x^2 + 3e^(-x)
    return 2*x - 3*np.exp(-x)
	
def GD(lr, x0):
    x = [x0]
    for iteration in range(100):
        x_new = x[-1] - lr * grad(x[-1])
        
        #Điều kiện dừng: khi độ lớn của đạo hàm đủ nhỏ
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, iteration) #Bao gồm các giá trị x và số vòng lặp để hội tụ
```
- **Khảo sát** với các điểm khởi tạo khác nhau, xác định:
	- Giá trị $x$;
	- Số vòng lặp để hội tụ.
```python
(x1, it1) = GD(0.1, 0) #lr = 0,1; x0 = 0
(x2, it2) = GD(0.1, 5) #lr = 0,1; x0 = 5

print('Solution x1 = %f, cost = %f, obtained after %d iterations' % (x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations' % (x2[-1], cost(x2[-1]), it2))
```
- **Khảo sát** với các tốc độ học ($\text{lr}$) khác nhau
```python
(x1, it1) = GD(0.1, 0)
(x2, it2) = GD(0.1, 5)

print('Solution x1 = %f, cost = %f, obtained after %d iterations' % (x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations' % (x2[-1], cost(x2[-1]), it2))
```
- **Điểm khởi tạo**:
    - Các điểm khởi tạo khác nhau có thể dẫn đến số vòng lặp khác nhau;
    - Thuật toán Gradient Descent thường hội tụ về nghiệm gần giống nhau khi hàm số có tính chất "đẹp".
- **Learning rate**:
	- Giá trị của $\text{lr}$ hay $\eta$ rất quan trọng.
	- Một giá trị quá nhỏ khiến quá trình hội tụ chậm, trong khi giá trị quá lớn có thể dẫn đến dao động hoặc phân kỳ.
# III. Gradient Descent cho hàm nhiều biến
## 1. Lý thuyết
- Gradient Descent không chỉ giới hạn ở hàm 1 biến mà còn rất *mạnh mẽ trong việc tối ưu các hàm số nhiều biến*.
- Trong Machine Learning, các hàm mất mát thường là hàm của nhiều biến (tham số mô hình).

- Ví dụ, **trong bài toán Linear Regression**, hàm mất mát có thể được viết dưới dạng:
  $$\mathcal{L}(\text{w}) = \dfrac{1}{2N}\lVert \text{y}-X\text{w} \lVert^2_2 = \dfrac{1}{2N}\sum_{i=1}^{N}{(\text{y}_i - \text{x}_i^T\text{w})^2}$$
  trong đó:
	- $\text{w}$: vector hệ số hồi quy - tham số mô hình;
	- $X$: ma trận đầu vào, mỗi hàng $\text{x}_i^T$ là vector đặc trưng của mẫu dữ liệu thứ $i$;
	- $y$: vector đầu ra thực tế;
	- Đạo hàm của hàm mất mát khi đó được xác định:
      $$\nabla_\text{w}\mathcal{L}(\text{w}) = \dfrac{1}{N}X^T(X\text{w}-\text{y})$$
	- Khi đó, tương tự như Hàm 1 biến, **Gradient Descent** cho hàm nhiều biến cập nhập vector tham số $\text{w}$ theo hướng ngược gradient:
      $$\text{w}_{t+1} = \text{w}_t - \eta\nabla_{\text{w}}\mathcal{L}(\text{w}_t)$$
      Quy tắc cập nhật này vẫn đảm bảo di chuyển $\text{w}$ về vùng có giá trị hàm mất mát nhỏ hơn.

- Giả sử ta cần đi tìm Global Minimum cho hàm nhiều biến $f(\theta)$ trong đó $\theta$ là một vector - kí hiệu được sử dụng để mô tả tập hợp các tham số của một mô hình mà thuật toán cần tối ưu.
- *Đạo hàm tại điểm $\theta$* bất kì được xác định là: $\nabla_\theta f(\theta)$.
- Tương tự với hàm một biến, ta cũng bắt đầu từ điểm dự đoán $\theta_0$, và **quy tắc cập nhập**:
  $$\theta_{t+1} = \theta_t - \eta\nabla_\theta f(\theta_t)$$
## 2. Đạo hàm
- Trong một số trường hợp, việc tính đạo hàm của hàm nhiều biến khá phức tạp. Ta có thể dựa trên định nghĩa đạo hàm (hàm 1 biến) để kiểm tra đạo hàm:
  $$f'(x) = \lim_{\varepsilon \to \infty}{\dfrac{ f(x + \varepsilon) - f(x) }{ \varepsilon }}$$
  Ta thường chọn $\varepsilon$ rất nhỏ và sử dụng
  $$f'(x) = \lim_{\varepsilon \to \infty}{\dfrac{ f(x + \varepsilon) - f(x - \varepsilon) }{ 2\varepsilon }}$$
## 3. Minh hoạ với Linear Regression
- Để minh hoạ, ta tạo dữ liệu giả định với đường thẳng $y = 3x+4$:
```python
X = np.random.rand(1000, 1)
y = 4 + 3*X + 0.2*np.random.randn(1000, 1)

one  = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1) #Thêm cột 1 (bias)
```
- Với bài toán này ta cần xác định Linear Regression với hai tham số: $w_0$ (bias) và $w_1$ (weight).
- Để đơn giản, ta thêm một cột bias vào ma trận $X$ để biễu diển bias; và khi đó $\text{w} = [w_0, w_1]^T$.
```python
def cost(w): #Lost function
	N = Xbar.shape[0]
	return 0.5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2
	
def grad(w):
	N = Xbar.shape[0]
	return (1/N)*Xbar.T.dot(Xbar.dot(w) - y)
```
# IV. Các thuật toán tối ưu Gradient Descent
## 1. Momentum
### 1.1 Ý tưởng
- Dự đoán điểm khởi tạo $\theta = \theta_0$.
- Cập nhập $\theta$ đến khi kết quả chấp nhận được:
  $$\theta = \theta - \eta\nabla_\theta J(\theta)$$
  ![](.\attachments\GDvsMomentumGD.png)
- Dựa vào hình ảnh ở trên, ta có thể thấy vấn đề với thuật toán GD khi xuất hiện một local minimum không mong muốn $\text{D}$.
- Tuy nhiên, bằng góc nhìn vật lý, khi viên bi $\text{B}$ lăn xuống tại $\text{D}$, nó vẫn có thể có đà và lăn ngược lên sau đó đi đến $\text{C}$.
- Dựa trên hiện tượng này, một thuật toán được ra đời nhằm khắc phục việc nghiệm của GD rơi vào một điểm local minimum không mong muốn. Thuật toán đó có tên là Momentum.
### 1.2 Momentum
- Thay vì trực tiếp đi ngược với đạo hàm sử dụng $\textcolor{lightblue}{\text{Learning rate}}$ và giá trị đạo hàm:
	- Ta tính một đại lượng gọi là *lượng thay đổi tại thời điểm $t$*;
	- Có thể coi như là **vận tốc** $v_t$ - nhằm tích lũy "đà - momentum" từ các bước cập nhập trước.
	- Khi đó, $v_t$ sẽ *mang 2 thông tin*:
		- Độ dốc - Gradient term (giá trị đạo hàm): $\eta\nabla_\theta J(\theta)$.
		- Đà - Momentum term (vận tốc trước đó): $v_{t-1}$.
	- Một cách đơn giản, ta sử dụng phép cộng có trọng số hai đại lượng này:
      $$v_t = \gamma v_{t-1} + \eta\nabla_\theta J(\theta)$$
      trong đó:
		- $\gamma$ (gamma) là *momentum coefficient*, $\gamma \in [0,1)$: mức độ duy trì vận tốc từ các bước trước đó (thường là $0.9$);
		- $\eta$: là learning rate - tốc độ học;
		- $\nabla_\theta J(\theta_t)$: gradient hàm mất mát tại $\theta_t$.

- Lúc này ta có **công thức cập nhập kết hợp Momentum**:
  $$\theta = \theta - v_t$$
  trong đó, dấu trừ thể hiện việc đi ngược đạo hàm.
## 3. Nesterov Accelarated Gradient (NAG)
### 3.1 Ý tưởng
- Thuật toán Momentum giúp hòn bi vượt qua dốc giá trị local minimum, tuy nhiên, khi đến gần đích, thuật toán sẽ mất thời gian để hội tụ do tính chất sử dụng "đà" của nó.
- **Nesterov Accelerated Gradient** (NAG) là một biến thể cải tiến của Momentum GD, nhằm tăng tốc độ hội tụ và khắc phục nhược điểm dao động khi gần điểm cực tiểu.
- **Ý tưởng chính**: "nhìn trước một bước" trong quá trình cập nhật vận tốc. Thay vì tính gradient tại vị trí hiện tại .

- Sử dụng Gradient xấp xỉ vị trí tiếp theo:
	- Sử dụng duy nhất Momentum: $\gamma v_{t-1}$ $\to$ xấp xỉ được vị trí tiếp theo.
	- Từ xấp xỉ, tính Gradient tiếp theo.
- Khi đó:
	- Momentum: _lượng thay đổi_ là tổng của Momentum vector và Gradient ở thời điểm hiện tại.
	- Nesterove momentum: _lượng thay đổi_ là tổng: Momentum vector và Gradient ở thời điểm xấp xỉ là điểm tiếp theo.
    ![](.\attachments\NesterovAcceleratedGradient.png)
- **Công thức cập nhập với Nesterove Momentum**:
  $$v_t = \gamma v_{t-1} + \eta\nabla_\theta J(\theta_t - \gamma v_{t-1})$$
  $$\theta_{t+1} = \theta_t - v_t$$
## 4. Các thuật toán khác
- Ngoài hai thuật toán trên, có rất nhiều thuật toán nâng cao khác được sử dụng trong các bài toán thực tế, đặc biệt là các bài toán Deep Learning:
	- Adagrad;
	- Adam;
	- RMSprop; ...
# V. Biến thể của Gradient Descent
- Sử dụng ví dụ đối với mô hình Linear Regression, ta có:
  $$\begin{align} J(\text{w}) &= \dfrac{1}{2N} \lVert \text{y} - \bar{X}\text{w} \lVert^2_2 \\[4pt] &= \dfrac{1}{2N} \sum_{i=1}^{N}{(\text{x}_i\text{w} - \text{y}_i)^2} \end{align}$$
  và Gradient
  $$\nabla_\text{w} J(\text{w}) = \dfrac{1}{N} \sum_{i=1}^{N}\text{x}_i^T(\text{x}_i\text{w}-\text{y}_i)$$
- Gradient Descent có ba biến thể chính, **khác nhau ở lượng dữ liệu sử dụng để tính gradient** trong mỗi lần cập nhật:
## 1. Batch Gradient Descent
 - Đây là thuật toán GD "thuần túy" được sử dụng từ ban đầu đến hiện tại.
 - Khi cập nhập $\theta$, ta sử dụng *Toàn bộ* các điểm dữ liệu $x_i$.
 - **Đánh giá**:
	 - Ưu điểm:
		 - Tính toán gradient chính xác, hội tụ ổn định.
	 - Nhược điểm:
		- Tốn thời gian tính toán cho mỗi lần cập nhật;
		- Không hiệu quả cho online learning (học trực tuyến) khi dữ liệu đến liên tục.
## 2. Stochastic Gradient Descent
- Tại một thời điểm, ta tính đạo hàm hàm mất mát *chỉ dựa trên một điểm* dữ liệu $x_i$ $\to$ cập nhập $\theta$.
- Mỗi lần toàn bộ dữ liệu được duyệt qua được gọi là một epoch,
	- Đối với Batch, mỗi epoch thì $\theta$ được cập nhập 1 lần.
	- Đối với Stochastic, mỗi epoch thì $\theta$ được cập nhập $N$ lần ($N$ là số điểm dữ liệu của tập huấn luyện)
- **Đánh giá**:
	- Ưu điểm:
		- Tính toán nhanh hơn nhiều so với BGD;
		- Phù hợp cho online learning;
		- Có thể thoát khỏi các điểm cực tiểu cục bộ nhờ độ nhiễu trong gradient.
	- Nhược điểm:
		- Gradient tính toán không chính xác, hội tụ không ổn định, có nhiều dao động.
		- Cần nhiều lần cập nhật hơn để hội tụ so với BGD.
### Lựa chọn điểm dữ liệu và Cập nhập
- Sau mỗi epoch, ta cần *shuffle* thứ tự để đảm bảo tính ngẫu nhiên và việc này cũng ảnh hưởng tới hiệu năng SGD.
- **Quy tắc cập nhập**:
  $$\theta = \theta - \eta\nabla_\theta J(\theta; x_i; y_i)$$
  trong đó ta chỉ tính đạo hàm hàm mất mát với một cặp điểm dữ liệu $(x_i, y_i)$.
- Ta cũng có thể áp dụng các thuật toán như Momentum, AdaGrad, ... vào SGD.
## 3. Mini-batch Gradient Descent
- Cách hoạt động tương tự SGD, tuy nhiên sử dụng *$n$ điểm dữ liệu* cho mỗi lần tính đạo hàm:
  $$\theta = \theta - \eta\nabla_\theta J(\theta; x_{i:i+n}; y_{i:i+n})$$
- **Đánh giá**:
	- Ưu điểm:
		- Cân bằng giữa tốc độ tính toán và độ ổn định hội tụ.
		- Phổ biến nhất trong Deep Learning vì hiệu quả trên thực tế.
		- Có thể tận dụng khả năng song song hóa của GPU để tính toán gradient trên mini-batch.
	- Nhược điểm:
		- Vẫn còn độ nhiễu trong gradient, nhưng ít hơn SGD.
		- Cần điều chỉnh kích thước mini-batch phù hợp.
# VI. Điều kiện dừng - Stopping Criteria
- Để thuật toán Gradient Descent dừng lại khi hội tụ, ta cần xác định stopping criteria, phổ biến là:
	1. **Giới hạn số vòng lặp (epoch)**:
		- Dừng thuật toán sau một số vòng lặp tối đa.
		- Là cách đơn giản và phổ biến nhất, nhưng có thể dừng quá sớm hoặc quá muộn.
	2. **So sánh gradient**:
		- Dừng khi độ lớn của gradient trở nên rất nhỏ (gần 0), cho thấy đã đến gần điểm cực tiểu.
		- Việc tính gradient có thể tốn kém, đặc biệt với tập dữ liệu lớn.
	3. **So sánh giá trị hàm mất mát**:
		- Dừng khi sự thay đổi giá trị hàm mất mát giữa các vòng lặp liên tiếp trở nên rất nhỏ $\to$ thuật toán đã hội tụ.
		- Có thể không hiệu quả nếu hàm mất mát có vùng bằng phẳng (saddle points).
	4. **Kết hợp với SGD và Mini-batch GD**:
		- So sánh nghiệm sau 1 vài lần cập nhật (ví dụ 10 lần trong SGD), và dừng khi sự thay đổi nghiệm trở nên rất nhỏ.
# VII. Phương pháp tối ưu: Newton’s method
- **Newton's method** (còn gọi là Newton-Raphson) là một phương pháp tối ưu bậc hai, sử dụng thông tin đạo hàm bậc hai (Hessian matrix) - độ cong hàm số, để tìm nghiệm của phương trình:
  $$f'(x) = 0$$
  từ đó tìm điểm cực tiểu của hàm số (nhanh hơn so với phương pháp bậc nhất như GD).
- Cụ thể:
	- Gradient Descent chỉ sử dụng đạo hàm bậc nhất (gradient) để xác định hướng đi xuống dốc nhất;
	- Newton's method tận dụng thêm đạo hàm bậc hai (Hessian matrix) để xấp xỉ hàm mục tiêu bằng một hàm bậc hai, từ đó tìm ra bước nhảy tối ưu hơn hướng đến điểm cực tiểu.
## 1. Newton’s method cho giải phương trình f(x)=0
- Newton's method ban đầu được phát triển là một thuật toán lặp để tìm nghiệm của phương trình $f(x)=0$.
- **Ý tưởng chính**:
	- Xấp xỉ hàm số $f(x)$ bằng đường tiếp tuyến tại điểm dự đoán hiện tại $x_t$;
	- Tìm giao điểm $x_{t+1}$ của đường tiếp tuyến với trục hoành - một ước lượng tốt hơn cho nghiệm hàm số.
	- Công thức cập nhật của Newton's method:
      $$x_{t+1} = x_t - \dfrac{f(x_t)}{f'(x_t)}$$
	- Trong đó thì $\left(-\dfrac{f(x_t)}{f'(x_t)}\right)$ là bước nhảy từ điểm hiện tại đến điểm tiếp theo.
	- Tỉ lệ trên phản ánh độ lớn giá trị hàm số / độ dốc của nó tại $x_t$, nghĩa là khi giá trị $f(x_t)$ nhỏ hoặc độ dốc $f'(x_t)$ nhỏ thì bước nhảy sẽ nhỏ (và ngược lại).
## 2. Newton’s method trong bài toán tìm local minimum
- Ap dụng Newton's method: bài toán tìm local minimum, cần giải phương trình $f'(x) = 0$ - cực tiểu cục bộ của $J(\theta)$
- Tức là thay $f(x)$ ở trên thành $f'(x)$, khi đó ta được **công thức cập nhập**:
  $$x_{t+1} = x_t - \dfrac{f'(x_t)}{f''(x_t)}$$
- Trong không gian nhiều chiều, công thức trên được viết thành:
  $$\theta_{t+1} = \theta_t - H(J(\theta_t))^{-1}\nabla_\theta J(\theta_t)$$
  trong đó:
	- $\nabla_\theta J(\theta_t)$: gradient hàm mất mát, vector đạo hàm bậc nhất chỉ hướng dốc nhất của hàm số tại điểm đó;
	- $H(J(\theta_t))$ là Hessian Matrix của hàm mất mát, ma trận vuông $n\times n$ ($n$ là số chiều $\theta$), chứa tất cả đạo hàm riêng bậc hai của hàm $J(\theta)$.
### Ma trận Hessian
- Ma trận Hessian là một công cụ quan trọng trong giải tích đa biến, đặc biệt là trong bài toán tối ưu hóa.
- Đối với hàm số vô hướng $J(\theta)$ của $n$ biến $\theta = [\theta_0, ..., \theta_n]^T$ , khi đó ta có:
  $$H(J(\theta)) = \begin{bmatrix} \dfrac{\partial^2 J}{\partial \theta_1^2} & \dfrac{\partial^2 J}{\partial \theta_1 \partial \theta_2} &\cdots & \dfrac{\partial^2 J}{\partial \theta_1 \partial \theta_n} \\ \vdots & \vdots & \ddots & \vdots \\ \dfrac{\partial^2 J}{\partial \theta_n \partial \theta_1} & \dfrac{\partial^2 J}{\partial \theta_n \partial \theta_2} &\cdots & \dfrac{\partial^2 J}{\partial \theta_n^2} \end{bmatrix}$$
- **Vai trò** trong Newton's method:
	- Thông tin độ cong: đo lường tốc độ thay đổi của đạo hàm bậc nhất - thông tin độ cong tại điểm đó.
	- Xấp xỉ bậc hai.
	- Xác định bước nhảy tối ưu: giúp hội tụ nhanh hơn.
- Áp dụng vào bài toán:
	- $−\nabla_\theta J(\theta_t)$: đảm bảo di chuyển ngược hướng gradient, tức là hướng xuống dốc.
	- $H(J(\theta_t​))^{−1}$: điều chỉnh bước nhảy dựa trên độ cong của hàm số.
		- Nếu độ cong lớn - tốc độ giảm gradient lớn (Hessian có trị riêng lớn), bước nhảy sẽ nhỏ hơn, và ngược lại.
		- Thích ứng với hình dạng của hàm số, tránh việc nhảy quá xa hoặc quá gần điểm cực tiểu.
## 3. Hạn chế của Newton's method
- **Điểm khởi tạo cần gần nghiệm**:
	- Newton's method hội tụ nhanh khi điểm khởi tạo ($\theta_0$) gần với nghiệm ($\theta^*$), nhưng có thể phân kỳ nếu điểm khởi tạo quá xa nghiệm.
- **Chi phí tính toán cao**:
	- Gradient $\nabla_\theta J(\theta_t)$;
	- Hessian matrix $H(J(\theta_t))$;
	- Ma trận nghịch đảo $H(J(\theta_t))^{−1}$: độ phức tạp tính toán của việc tính ma trận nghịch đảo thường là $O(n^3)$.
- **Hessian matrix không là ma trận xác định dương**, để Newton's method hội tụ, Hessian matrix $H(J(\theta_t))$ cần phải là ma trận xác định dương (positive definite) trong quá trình lặp. Nếu Hessian matrix không xác định dương, phương pháp có thể di chuyển theo hướng không mong muốn (như tới điểm cực đại, điểm yên ngựa)
- **Hessian matrix suy biến** (singular), tức là có $\text{Det} = 0$: ma trận nghịch đảo không tồn tại.
## 4. So sánh với Gradient Descent

| Đặc điểm              | Newton's Method                                   | Gradient Descent                               |
| --------------------- | ------------------------------------------------- | ---------------------------------------------- |
| **Bậc đạo hàm**       | Bậc hai (Hessian matrix)                          | Bậc nhất (Gradient)                            |
| **Tốc độ hội tụ**     | Nhanh hơn (hội tụ bậc hai, quadratic convergence) | Chậm hơn (hội tụ bậc nhất, linear convergence) |
| **Chi phí tính toán** | Cao (tính Hessian và nghịch đảo)                  | Thấp (chỉ tính gradient)                       |
| **Độ ổn định**        | Kém ổn địnn, dễ phân kỳ nếu khởi tạo xa nghiệm    | Ổn định hơn, ít bị phân kỳ                     |
| **Yêu cầu**           | Hàm số phải khả vi hai lần                        | Hàm số phải khả vi một lần                     |
# VIII. Kết luận
- Gradient Descent là thuật toán tối ưu mạnh mẽ và linh hoạt, được sử dụng trong cả Machine Learning và Deep Learning. 
- Mặc dù có những hạn chế nhất định, Gradient Descent và các biến thể của nó vẫn là công cụ không thể thiếu để huấn luyện các mô hình học máy hiệu quả.
- Việc lựa chọn biến thể GD phù hợp (BGD, SGD, MBGD), thuật toán tối ưu (Momentum, NAG, Adam, ...) và điều chỉnh các hyperparameters (learning rate, momentum coefficient, ...) là rất quan trọng để đạt được hiệu suất tốt nhất trong từng bài toán cụ thể.

# Tham khảo
[1] IBM. [What is gradient descent?](https://www.ibm.com/topics/gradient-descent)

[2] Khan Academy. [Gradient descent (article)](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/optimizing-multivariable-functions/a/what-is-gradient-descent)

[3] Machine Learning cơ bản. Bài 7: Gradient Descent (phần 1/2) - [link](machinelearningcoban.com/2017/01/12/gradientdescent)

[4] Machine Learning cơ bản. Bài 8: Gradient Descent (phần 2/2) - [link](machinelearningcoban.com/2017/01/16/gradientdescent2)