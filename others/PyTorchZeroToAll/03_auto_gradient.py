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
        # ����������������������batch�ڵ��ݶȼ��㣬��batch�ڵ��������ϵ��ݶ���͡�
        # ��������������������������ÿ��batch������һ����Ҫ�ݶ���Ϊ0����batch֮����ݶȻ��໥Ӱ��
        w.grad.data.zero_()

    print("progress:", epoch, l.data[0])

# After training
print("predict (after training)", 4, forward(4).data[0])
