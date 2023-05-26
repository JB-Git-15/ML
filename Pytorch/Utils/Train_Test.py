def train_model(n_epoch:int, n_samples:int, n_classes:int, model: nn.Module, loss_fn: nn.modules.loss, optimizer:torch.optim, train_data_loader: DataLoader, device: torch.device):
    
    model.to(device)
    model.train()
    loss_train, acc = 0, 0
    for (X,y) in train_data_loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss   = loss_fn(y_pred, nn.functional.one_hot(y, num_classes= n_classes).type(torch.float32))
        acc    = acc + (y_pred.argmax(dim=1) == y).sum()
        loss_train = loss_train + loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (n_epoch % n_samples)==0:        
        loss_train = loss_train/len(train_data_loader)
        acc = acc/(len(train_data_loader)*train_data_loader.batch_size)
        return loss_train.detach().item(), acc.detach().item()
    else:
        return np.nan, np.nan      



def test_model(n_epoch:int, n_samples:int, n_classes:int , model: nn.Module, loss_fn: nn.modules.loss, test_data_loader: DataLoader, device: torch.device):

    loss_train, acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for (X,y) in test_data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss   = loss_fn(y_pred, nn.functional.one_hot(y, num_classes= n_classes).type(torch.float32))
            acc    = acc + (y_pred.argmax(dim=1) == y).sum()
            loss_train = loss_train + loss

        if (n_epoch % n_samples)==0:        
            loss_train = loss_train/len(test_data_loader)
            acc = acc/(len(test_data_loader)*test_data_loader.batch_size)
            return loss_train.detach().item(), acc.detach().item()
        else:
            return np.nan, np.nan         



def train_test_model(n_epochs:int, n_samples:int, n_classes:int, model:nn.Module, loss_function:nn.modules.loss, optimizer:torch.optim, train_data_loader: DataLoader, test_data_loader: DataLoader, device: torch.device):
    """
    n_epochs: number of passes for all the training data
    n_samples: one every n_samples, save the loss and the accuracy. 
    n_classes: number of classes
    model: model to test
    loss_function: loss function (can be BCELoss for instance)
    optimizer: (for instance nn.) 
    """

    loss_train_epochs = []
    loss_test_epochs  = []
    acc_train_epochs  = []
    acc_test_epochs   = []
    num_epoch         = []

    for n_epoch in tqdm(range(n_epochs)):

        loss_temp_train, acc_temp_train = train_model(n_epoch, n_samples, n_classes, model, loss_function, optimizer, train_data_loader, device)
        loss_temp_test, acc_temp_test = test_model(n_epoch, n_samples, n_classes , model, loss_function, test_data_loader, device)

        if ((n_epoch %n_samples) == 0):
            loss_train_epochs.append(loss_temp_train)
            loss_test_epochs.append(loss_temp_test)
            acc_train_epochs.append(acc_temp_train)
            acc_test_epochs.append(acc_temp_test)
            num_epoch.append(n_epoch)

    return loss_train_epochs, acc_train_epochs, loss_test_epochs, acc_test_epochs, num_epoch





def prediction(model, test_data_loader, device: torch.device):
    model.to(device)
    model.eval()
    list_gt = []
    list_pred = []
    with torch.inference_mode():
        for (X,y) in test_data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X).argmax(dim=1)
            
            list_gt.append(y)
            list_pred.append(y_pred)
    
    return list_gt, list_pred


def plot_results(num_epoch, loss_train_epochs, acc_train_epochs, loss_test_epochs, acc_test_epochs):

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,5))
    ax1.plot(num_epoch, loss_train_epochs, label = 'train')
    ax1.plot(num_epoch, loss_test_epochs, label= 'test')
    ax1.set_xlabel('# epoch')
    ax1.set_ylabel('loss')
    ax1.legend()
 
    ax2.plot(num_epoch, acc_train_epochs, label = 'train')
    ax2.plot(num_epoch, acc_test_epochs, label= 'test')
    ax2.set_xlabel('# epoch')
    ax2.set_ylabel('accuracy')
    ax2.legend()


def confusion_matrix( list_gt, list_pred, classes):
    # vertical dimension: GT
    # horizontal dimension: Predicted
    
    n_classes = len(classes)
    CM = np.zeros((len(classes),len(classes)))
    for (l_gt, l_pred) in zip(list_gt, list_pred):
      for (gt, pred) in zip(l_gt, l_pred):
        CM[gt.item(), pred.item()] += 1

    fig, ax = plt.subplots()
    ax.imshow(CM, cmap= 'spring')
    ax.set_xticks(np.arange(n_classes),classes, rotation=45);
    ax.set_yticks(np.arange(n_classes),classes, rotation=0);
    for (j,i),label in np.ndenumerate(CM):
        ax.text(i,j,int(label),ha='center',va='center')


















