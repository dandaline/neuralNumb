# neuralNumb

A 3-Layer neural network to detect written digits
thanks to Samson Zhang for teching me how this works
## Dataset ##

MNIST train and test dataset from http://yann.lecun.com/exdb/mnist/ formatted into csv by https://pjreddie.com/projects/mnist-in-csv/

Each image of the training dataset is 28x28 pixels and each pixel has a value between 0 and 255. A 0 means that the pixel is white, a 255 means that the pixel is black. The values in between are different shades of grey.

The first column of the dataset is the label of the image. A label is the number that is written on the image. The following 784 columns are the pixel values (a number between 0 and 255).



## Model ##

**Forward propagation**

zero layer = input vector = 28*28 * 0 = 784 * 0 <br>
$A_0 = X$<br>

first layer - middle <br>
$Z_1 = W_1 A_0 + b_1$<br>
$A_1 =  \forall z \in Z¹: ReLu(z_i) =
                                    \begin{cases}
                                    z & \text{if } z_m > 0 \\
                                    0 & \text{if } z_m \leq 0
                                    \end{cases}$ <br>

third layer - output <br>
$Z_2 = W_2 A_1 + b_2$<br>
$A_2 = \forall z \in Z_2: SoftMax(z_i) = \frac{e^{z_i}}{\sum\limits_{i=0}^{m}e^{z_i}}$

**Backward propagation**

Resulting predictions - correct prediction = error of the second layer <br>
$\Delta Z_2 = A_2 - Y$ <br>
$Y := \forall y \in Y: 
                        \begin{cases}
                        1 & \text{if } y_m = \max(Y) \\
                        0 & \text{if } y_m \ne \max(Y)
                        \end{cases}$ <br>

calculate error contribution of weight 2 <br>
$\Delta W_2 = \frac{1}{m} * \Delta Z_2 * A_1^T$<br>

calculate error contribution of bias 2 <br>
$\Delta b_2 = \frac{1}{m}{\sum\limits_{i=0}^{m}{\Delta Z_2}}$<br>

first layer - resulting prediction * the derivative of the ReLu-function = error of the first layer<br>
$\Delta Z_1 = W_2^T * \Delta Z_2 * ReLu'(Z_1)$<br>

calculate error contribution of weight 1 <br>
$\Delta W_1 = \frac{1}{m}\Delta Z_1 * X^T$<br>

calculate error contribution of bias 1 <br>
$\Delta b_1= \frac{1}{m}\sum\limits_{i=0}^{m}\Delta Z_1$

**Nudge the model**<br>

$W_1 := W_1 - \alpha * \Delta W_1$<br>
$b_1 := b_1 - \alpha * \Delta b_1$<br>
$W_2 := W_2 - \alpha * \Delta W_2$<br>
$b_2 := b_2 - \alpha * \Delta b_2$<br>
$\alpha$ is the learning factor - you set this to define, how dramatic the model should apply the changes