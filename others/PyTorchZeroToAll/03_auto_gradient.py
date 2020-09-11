import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)  # Any random value


# our model forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# Before training
print("predict (before training)", 4, forward(4).data[0])

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        # compute the loss in forward
        l = loss(x_val, y_val)

        # compute the gradients
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])

        # update the gradients, like optimizer.step() does
        w.data = w.data - 0.01 * w.grad.data

        # Manually zero the gradients after updating weights, like optimizer.zero_grad() does
        # ！！！！！！！！！！！batch内的梯度计算，是batch内的所有语料的梯度求和。
        # ！！！！！！！！！！！所以每个batch处理完一次需要梯度置为0，否定batch之间的梯度会相互影响
        w.grad.data.zero_()

    print("progress:", epoch, l.data[0])

# After training
print("predict (after training)", 4, forward(4).data[0])
