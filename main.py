from VGG16 import VGGNet
from imports import *
from utils import *
from config import *


def load_dataset(PATH):
	train_set = VGGLoader(PATH)
	train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
	return train_loader


def train (train_loader, optimizer, model):
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    train_loss = []

    for iteration, batch in (enumerate(tqdm.auto.tqdm(train_loader))):
        optimizer.zero_grad()
        input, label = batch
        predict = model (input)
        predict = predict.cuda()

        mse = mse_loss (predict, label)
        train_loss.append(mse)

        mse.backward()
        optimizer.step()    
    model.eval()

    return mse


    ################ VALIDARE ###################
    #model.train()

        
def main():
    model = VGGNet()
    print('########################')
    print('Training has started...')
    if MODEL_PATH == None:
        start_epoch = 1
        print(f'Untrained model, starting epoch: {start_epoch}')
    else:
        ckpt = torch.load(MODEL_PATH)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        start_epoch = ckpt['epoch'] + 1
        print(f'Trained model, starting epoch: {start_epoch} from: {MODEL_PATH}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    train_loader = load_dataset(PATH)
    optimizer = torch.optim.Adam(list(model.parameters()), lr = 10**-(5))
    epoch_loss = []

    for epoch in tqdm.tqdm (range(start_epoch, start_epoch + number_of_epochs)): 
        mse = train (train_loader, optimizer, model)
        print('Epoch:', epoch, 'has mse:', mse)
        print('########################')
        epoch_loss.append(mse)

        torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': mse}, 
                        SAVE_MODEL_PATH)
    

if __name__ == '__main__':
    main()
