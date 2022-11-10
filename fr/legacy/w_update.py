"""
	https://www.educba.com/pytorch-backward/
	https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
"""
from torch import optim
import torch

X = torch.tensor([[2,4,6],
                  [2,7,9],
                  [2,4,5]], dtype=torch.float, requires_grad=True)
lr = 0.01
optimizer = optim.SGD([X], lr=lr, weight_decay=0.0, momentum=0.8)
for i in range(2):
	for j in range(3):
		# x = X[i, j].clone().detach()
		x = X[i,j].detach().numpy().item()
		output = X[i,j] * X[i,j]
		# print('before: ', X.grad[i, j])
		optimizer.zero_grad()
		output.backward()       # compute the gradient for each parameter
		print('after: ', X.grad[i,j])
		optimizer.step()        # Update: x = x - \eta * dloss/dx
		print(X[i,j], x - lr * X.grad[i,j])
print(X)

# import torch
# A = torch.tensor(2.0, requires_grad = True)
# B = A * 4
# B.backward()
#
# print(A.grad.data)

#
# X = torch.tensor([[2,4,6],
#                   [2,7,9],
#                   [2,4,5]], dtype=torch.float, requires_grad=True)
#
# # X2 = torch.tensor([[1,2,3],
# #                   [4, 5, 6],
# #                   [7, 8, 9]], dtype=torch.float, requires_grad=True)
# X2 = 3 * X
#
# lr = 0.01
# optimizer = optim.SGD([X], lr=lr, weight_decay=0.0)
# for i in range(2):
# 	for j in range(3):
# 		# x = X[i, j].clone().detach()
# 		x = X[i,j].detach().numpy().item()
# 		x2 = X2[i, j].detach().numpy().item()
# 		output = X[i,j] * X[i,j]    # first layer
# 		output = X2[i, j] + output  # second layer
# 		# print('before: ', X.grad[i, j])
# 		optimizer.zero_grad()
# 		output.backward()       # compute the gradient for each parameter
# 		print('after: ', X.grad[i,j], X2.grad[i,j])
# 		optimizer.step()        # Update: x = x - \eta * dloss/dx
# 		print(X[i,j], x - lr * X.grad[i,j], X2[i,j], x2-lr * X2.grad[i,j])
# print(X)
# print(X2)


# -*- coding: utf-8 -*-
import torch

# Create Tensors to hold input and outputs.
# x = torch.linspace(-math.pi, math.pi, 2000)
# y = torch.sin(x)

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).
# p = torch.tensor([1, 2, 3])
x = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.float)
# xx = x.unsqueeze(-1)
xx = x
y = torch.tensor([1, 2], dtype=torch.float)

# In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
# (3,), for this case, broadcasting semantics will apply to obtain a tensor
# of shape (2000, 3)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. The Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# The Flatten layer flatens the output of the linear layer to a 1D tensor,
# to match the shape of `y`.
model = torch.nn.Sequential(
	torch.nn.Linear(3, 3, bias=False),
	torch.nn.Linear(3, 1, bias=False),
	# torch.nn.Flatten(0, 1)
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):

	# Forward pass: compute predicted y by passing x to the model. Module objects
	# override the __call__ operator so you can call them like functions. When
	# doing so you pass a Tensor of input data to the Module and it produces
	# a Tensor of output data.
	y_pred = model(xx)

	# Compute and print loss. We pass Tensors containing the predicted and true
	# values of y, and the loss function returns a Tensor containing the
	# loss.
	# loss = loss_fn(y_pred, y)
	pm = list(model.parameters())
	print(xx @ pm[0].T @ pm[1].T)
	print(y_pred)
	loss = torch.sum(y_pred)
	print(loss)
	if t % 100 == 99:
		print(t, loss.item())

	# for param in model.parameters():
	# 	print(param)
	# Zero the gradients before running the backward pass.
	model.zero_grad()

	# Backward pass: compute gradient of the loss with respect to all the learnable
	# parameters of the model. Internally, the parameters of each Module are stored
	# in Tensors with requires_grad=True, so this call will compute gradients for
	# all learnable parameters in the model.
	loss.backward()
	y = 0
	x = xx
	for param in model.parameters():
		print(param, param.grad)
	# 	x = x @ param
	# 	y += x
	# 	print(param, x, y)

	print(y)

	# Update the weights using gradient descent. Each parameter is a Tensor, so
	# we can access its gradients like we did before.
	with torch.no_grad():
		for param in model.parameters():
			param -= learning_rate * param.grad

# You can access the first layer of `model` like accessing the first item of a list
linear_layer = model[0]

# For linear layer, its parameters are stored as `weight` and `bias`.
print(
	f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
