# python 3.11


import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time


## math function to approximate (suggested by ChatGPT):
# f(x) = sum over i of x_i * sin(x_i) + 0.5 * sum over i of x_i^2


device = "mlx"
# device = torch.device("mps")
# device = torch.device("cuda")

print(f"using device: {device}")


num_samples = 50000
input_dim = 1000
hidden_dim = 4096
output_dim = 1 
batch_size = 128
epochs = 5
learning_rate = 0.001



# f(x) = sum over i of x_i * sin(x_i) + 0.5 * sum over i of x_i^2
X_train = mx.random.normal((num_samples, input_dim))

y_train = (
    mx.sum(X_train * mx.sin(X_train), axis=1)
    + 0.5 * mx.sum(X_train**2, axis=1)
)

y_train = mx.expand_dims(y_train, axis=1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        x = mx.maximum(self.fc1(x), 0)
        x = mx.maximum(self.fc2(x), 0)
        x = self.fc3(x)
        return x


net = Net()


def loss_fn(model, x, y):
    pred = model(x)
    return mx.mean((pred - y) ** 2)


loss_and_grad_fn = nn.value_and_grad(net, loss_fn)

optimizer = optim.Adam(learning_rate=learning_rate)


print("training...")
start_time = time.time()

for epoch in range(epochs):

    perm = mx.random.permutation(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]

    epoch_loss = 0.0

    for i in range(0, X_train.shape[0], batch_size):

        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        loss, grads = loss_and_grad_fn(net, batch_X, batch_y)

        optimizer.update(net, grads)

        mx.eval(net.parameters(), optimizer.state)

        epoch_loss += loss.item()

    avg_loss = epoch_loss / (X_train.shape[0] / batch_size)

    print(f"epoch: {epoch+1},   loss: {avg_loss:.4f}")

end_time = time.time()

print(f"runtime: {end_time - start_time:.2f} seconds")