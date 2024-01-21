import sys
sys.path.append("/workspace/TalkingStyle")
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import torch
import torch.nn as nn
from data_loader import get_dataloaders
from talkingstyle import TalkingStyle

def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=100):
    dev = args.device
    save_path = os.path.join(args.dataset,args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    min_loss = 1000

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0
    for e in range(epoch+1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template, one_hot  = audio.to(dev), vertice.to(dev), template.to(dev), one_hot.to(dev)
            loss, loss_detail = model(audio, template, vertice, one_hot, criterion, teacher_forcing=False)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.8f}".format((e+1), iteration, np.mean(loss_log)))


        # validation
        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all,file_name in dev_loader:
            # to gpu
            audio, vertice, template, one_hot_all= audio.to(dev), vertice.to(dev), template.to(dev), one_hot_all.to(dev)
            train_subject = "_".join(file_name[0].split("_")[:-1])
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:,iter,:]
                loss, loss_detail = model(audio, template, vertice, one_hot, criterion, teacher_forcing=False)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:,iter,:]
                    loss = model(audio, template,  vertice, one_hot, criterion)
                    valid_loss_log.append(loss.item())
                        
        current_loss = np.mean(valid_loss_log)

        if save_path is None:
            os.makedirs(save_path)

        if current_loss < min_loss:
            print("saving best model...")
            print("current loss: ", current_loss)
            min_loss = current_loss
            torch.save(model.state_dict(), os.path.join(save_path,'best_model.pth'))

        if (e > 0 and e % 5 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))

        print("epcoh: {}, current loss:{:.7f}".format(e+1,current_loss))
    return model

@torch.no_grad()
def test(args, model, test_loader,epoch):
    dev = args.device

    result_path = os.path.join(args.dataset,args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.dataset,args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(args.device)
    model.eval()
   
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all= audio.to(dev), vertice.to(dev), template.to(dev), one_hot_all.to(dev)
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze() # (seq_len, V*3)
                np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='TalkingStyle: A Novel Approach for the Generation of Personalized and Stylized Talking Face Avatars')
    parser.add_argument("--lr", type=float, default=0.0002, help='learning rate')
    parser.add_argument("--dataset", type=str, default="BIWI", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=23370*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=128, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "/workspace/CodeTalker_first_test/BIWI/wav_all", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="/workspace/CodeTalker_first_test/BIWI/vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save/models", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="save/result", help='path to the predictions')
    parser.add_argument("--lip_region", type=str, default="BIWI/regions/lip.txt", help='path to the lip region')
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--val_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    # parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
    #    " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
    #    " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
    #    " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    # parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
    #    " FaceTalk_170908_03277_TA")
    # parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
    #    " FaceTalk_170731_00024_TA")
    args = parser.parse_args()

    #build model
    model = TalkingStyle(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(args.device)
    
    #load data
    dataset = get_dataloaders(args)

    # loss
    criterion = nn.MSELoss()

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    model = trainer(args, dataset["train"], dataset["test"], model, optimizer, criterion, epoch=args.max_epoch)
    
    test(args, model, dataset["test"], epoch=args.max_epoch)
    
if __name__=="__main__":
    main()