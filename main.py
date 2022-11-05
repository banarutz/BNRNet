from VGG16 import VGGNet, ECCVGenerator
from imports import *
from utils import *
from config import *


def load_dataset(PATH):
	train_set = VGGLoader(PATH)
	train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
	return train_loader

def validation_dataset(VALIDATION_PATH):
    train_set = VGGLoader(VALIDATION_PATH)
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    return train_loader


def train (train_loader, valid_loader, optimizer, model, epoch):
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    train_loss = []
    print('### Validation ###')
    for iteration, batch in (enumerate(tqdm.auto.tqdm(train_loader))):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        input, label = batch
        input, label = input.to("cuda"), label.to("cuda")
        predict = model(input)

        MSE = torch.nn.L1Loss ()
        mse = MSE (predict, label)

        mse.backward()
        optimizer.step()
        mse = to_numpy(mse)
        train_loss.append(mse)

        #####DETACH AICI################
        #  



 ################ VALIDARE ###################
    model.eval()
    mae_valid_loss =[]
    mse_valid_loss =[]
    with torch.no_grad():
        for iteration, batch in (enumerate(tqdm.auto.tqdm(train_loader))):
            input, label = batch
            input, label = input.to("cuda"), label.to("cuda")
            predict = model(input)

            MAE = torch.nn.L1Loss ()
            mae = MAE (predict, label)
            mae = to_numpy(mae)
            mae_valid_loss.append(mae)

            MSE = torch.nn.MSELoss ()
            mse = MSE (predict, label)
            mse = to_numpy(mse)
            mse_valid_loss.append(mse)

    #     print('am ajuns aici')
    return np.mean(train_loss), np.mean(mse_valid_loss), np.mean(mae_valid_loss)


   
    #model.train()

        
def main():
    model = VGGNet()
    # model = ECCVGenerator()
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
    valid_loader = load_dataset(VALIDATION_PATH)
    optimizer = torch.optim.Adam(list(model.parameters()), lr = 10**-(5))

    writer = SummaryWriter()
    for epoch in tqdm.tqdm (range(start_epoch, start_epoch + number_of_epochs)): 
        train_loss, mse_valid, mae_valid = train (train_loader, valid_loader, optimizer, model, epoch)

        writer.add_scalar('Train loss -> MSE', train_loss, epoch)
        writer.add_scalar('MAE validation', mae_valid, epoch)
        writer.add_scalar('MSE validation', mse_valid, epoch)

        print('Epoch:', epoch, 'has mae:', train_loss, 'valid_mae:', mae_valid, 'valid_mse:', mse_valid)
        print('########################')
        # epoch_loss.append(mse)

        if not os.path.exists(SAVE_MODEL_PATH):
            os.makedirs(SAVE_MODEL_PATH)

        model_path = os.path.join(SAVE_MODEL_PATH, f'{str(epoch)}.pth')
        torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss}, 
                        model_path)
    

if __name__ == '__main__':
    main()
