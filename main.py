from VGG16 import VGGNet, ECCVGenerator
from imports import *
from utils import *
from config import *


def load_dataset(PATH):
	train_set = VGGLoader(PATH)
	train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
	return train_loader


def train (train_loader, optimizer, model, epoch):
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    train_loss = []
    
    for iteration, batch in (enumerate(tqdm.auto.tqdm(train_loader))):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        input, label = batch
        input, label = input.to("cuda"), label.to("cuda")
        predict = model(input)
        # predict = predict.cuda()
        # label = label.cuda()

        # predict = to_numpy(predict)
        # label = to_numpy(label)

        # print(label.shape)

        MSE = torch.nn.L1Loss ()
        mse = MSE (predict, label)
        # mse = get_mse(predict[:,0,:,:] , label[:,0,:,:])
        # mse += get_mse(predict[:,1,:,:] , label[:,1,:,:]) 
        # mse = np.array(mse)
        # mse = torch.tensor(mse, requires_grad = True)
        # mse = Variable(mse, requires_grad = True)
        mse.backward()
        optimizer.step()
        mse = to_numpy(mse)
        train_loss.append(mse)

        #####DETACH AICI################
        #  

    # model.eval()
    # with torch.no_grad():
    #     print('am ajuns aici')

 ################ VALIDARE ###################

    # return np.mean(train_loss)
    return np.mean(train_loss)


   
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
    optimizer = torch.optim.Adam(list(model.parameters()), lr = 10**-(5))
    epoch_loss = []
    writer = SummaryWriter()
    for epoch in tqdm.tqdm (range(start_epoch, start_epoch + number_of_epochs)): 
        mse = train (train_loader, optimizer, model, epoch)
        writer.add_scalar('Train loss -> MSE', mse, epoch)

        print('Epoch:', epoch, 'has mse:', mse)
        print('########################')
        # epoch_loss.append(mse)

        if not os.path.exists(SAVE_MODEL_PATH):
            os.makedirs(SAVE_MODEL_PATH)

        model_path = os.path.join(SAVE_MODEL_PATH, f'{str(epoch)}.pth')
        torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': mse}, 
                        model_path)
    

if __name__ == '__main__':
    main()
