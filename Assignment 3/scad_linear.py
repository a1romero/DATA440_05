import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cpu')
dtype = torch.float64

class SCADLinear(nn.Module):
    def __init__(self, input_size, lambda_val, a_val):
        super(SCADLinear, self).__init__() # init nn.Module
        self.input_size = input_size
        self.lambda_val = lambda_val
        self.a_val = a_val

        self.linear = nn.Linear(input_size, 1, bias= False, device= device, dtype= dtype)

    def forward(self, x): # how necessary is this?
        # train the linear model on x
        return self.linear(x)
    
    def scad_derivative(self, beta_hat):
        solution = self.lambda_val*((beta_hat <= self.lambda_val) + (self.a_val*self.lambda_val-beta_hat)*((self.a_val*self.lambda_val-beta_hat) > 0)/((self.a_val-1)*self.lambda_val)*(beta_hat>self.lambda_val))
        return solution
    
    def loss_with_scad(self, y_pred, y_true):
        mse = nn.MSELoss()(y_pred,y_true)
        penalty = torch.squeeze(self.scad_derivative(beta_hat=self.linear.weight))[1]
        return mse + penalty
    
    def fit(self, x, y, num_epochs= 200, learning_rate= .001):
        optimize = optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()
            optimize.zero_grad() # clear the optimizer
            y_pred = self(x)
            loss_with_scad = self.loss_with_scad(y_pred, y)
            loss_with_scad.backward() # computes gradient
            optimize.step()

            if (epoch+1) % 100 == 0:
                print(f'epoch: {epoch+1}/{num_epochs}, loss_with_scad: {loss_with_scad.item()}')
        return
    
    def predict(self, x):
        self.eval()
        with torch.no_grad(): # forces custom gradient
            y_pred = self(x)
        return y_pred
    
    def get_coefficients(self):
        return self.linear.weight