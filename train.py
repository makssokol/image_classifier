import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from .workspace_utils import active_session

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
checkpoint_file = 'checkpoint.pth'

training_transform = transforms.Compose([transforms.RandomRotation(33),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


testing_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

validation_transform = testing_transform

train_image_dataset = datasets.ImageFolder(train_dir, transform=training_transform)
test_image_dataset = datasets.ImageFolder(test_dir, transform=testing_transform)
valid_image_dataset = datasets.ImageFolder(valid_dir, transform=validation_transform)

trainloader = torch.utils.data.DataLoader(train_image_dataset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_image_dataset, batch_size=32)
validationloader = torch.utils.data.DataLoader(valid_image_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = models.densenet161(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(2208, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device)

epochs = 3
steps = 0
running_loss = 0
print_every = 10

with active_session():
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        valid_loss = criterion(logps, labels)
                        validation_loss += valid_loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f'Epoch {epoch+1}/{epochs}..'
                      f'Train loss: {running_loss/print_every:.3f}..'
                      f'Validation loss: {validation_loss/len(validationloader):.3f}'
                      f'Test accuracy: {accuracy/len(validationloader):.3f}')
                running_loss = 0
                model.train()

def model_testing(model, testloader):
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Test accuracy: {accuracy / len(testloader):.3f}")



def save_checkpoint(model, train_image_dataset, optimizer, checkpoint_file):
    model.class_to_idx = train_image_dataset.class_to_idx
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'classifier_state_dict': model.classifier.state_dict(),
                  'epochs': 3}

    torch.save(checkpoint, checkpoint_file)
