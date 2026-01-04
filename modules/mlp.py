from backend import xp
from .activations import gelu, gelu_derivative

class MLP:
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.0):

        scale = xp.sqrt(2.0 / input_dim)
        self.W_fc = xp.random.randn(input_dim, hidden_dim) * scale
        self.b_fc = xp.zeros(hidden_dim)

        self.W_proj = xp.random.randn(hidden_dim, input_dim) * scale
        self.b_proj = xp.zeros(input_dim)
        
        self.dropout_rate = dropout_rate
        self.params = [self.W_fc, self.b_fc, self.W_proj, self.b_proj]
        self.training = True
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True
    
    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        self.input = x
        
        self.hidden = xp.matmul(x, self.W_fc) + self.b_fc
        
        self.activated = gelu(self.hidden)
        
        self.output = xp.matmul(self.activated, self.W_proj) + self.b_proj
        
        if self.dropout_rate > 0 and self.training:
            self.dropout_mask = (xp.random.rand(*self.output.shape) > self.dropout_rate).astype(float)
            scale = 1.0 / (1.0 - self.dropout_rate)
            self.output = self.output * self.dropout_mask * scale
        
        return self.output

    def backward(self, gradient):
        if self.dropout_rate > 0 and self.training:
            scale = 1.0 / (1.0 - self.dropout_rate)
            gradient = gradient * self.dropout_mask * scale

        batch_dims = tuple(range(gradient.ndim - 1))
        activated_flat = self.activated.reshape(-1, self.activated.shape[-1])
        gradient_flat = gradient.reshape(-1, gradient.shape[-1])
        dW_proj = xp.matmul(activated_flat.T, gradient_flat)
        
        db_proj = xp.sum(gradient, axis=batch_dims)
        
        dactivated = xp.matmul(gradient, self.W_proj.T)
        
        dhidden = dactivated * gelu_derivative(self.hidden)
        
        input_flat = self.input.reshape(-1, self.input.shape[-1])
        dhidden_flat = dhidden.reshape(-1, dhidden.shape[-1])
        dW_fc = xp.matmul(input_flat.T, dhidden_flat)
        
        db_fc = xp.sum(dhidden, axis=batch_dims)
        
        dinput = xp.matmul(dhidden, self.W_fc.T)
        
        self.grads = [dW_fc, db_fc, dW_proj, db_proj]
        
        return dinput
