# Import dependencies
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.dataloader import Dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get data
train = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
dataset = Dataloader(train, 32) # A batches of 32 images

# Image Classifier Class
class ImageClassifier(nn.module):
    def __init__(self):
        super().__init__()
        self.model = nn.sequential(
            nn.conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )
    def forward(self, x ):
        return self.model(x)
    
# Instance of the neural network, loss, optimizer
classifier = ImageClassifier().to('cuda')
optimizer = Adam(classifier.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

# Training flow
if __name__=="main":
    for epoch in range(10): # Run for 10 epochs
        for batch in dataset:
             X,y = batch
             X,y = X.to('cuda'), y.to('cuda')
             y_hat = classifier(X)
             loss = loss_function(y_hat, y)

             # Applying backpropagation
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()

        print(f"Epoch:{epoch} loss is {loss.item()}")

    with open('model_state.pt', 'wb') as f:
        save(classifier.state_dict(), f)
        