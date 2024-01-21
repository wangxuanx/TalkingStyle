import pickle
import sys
sys.path.append("/workspace/TalkingStyle")
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import torch
from talkingstyle import TalkingStyle
from transformers import AutoProcessor
import librosa

@torch.no_grad()
def test(args, model, wav, path, epoch):
    dev = args.device

    result_path = os.path.join(args.dataset, args.result_path)
    # if os.path.exists(result_path):
    #     shutil.rmtree(result_path)
    # os.makedirs(result_path)

    wav_path = os.path.join(path, wav)
    processor = AutoProcessor.from_pretrained("/workspace/hubert-large-ls960-ft")
    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
    audio = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)

    save_path = os.path.join(args.dataset, args.save_path)
    test_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(args.device)
    model.eval()

    one_hot_all = np.eye(len(args.train_subjects.split(' ')))

    template_file = os.path.join(args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    template = templates['FaceTalk_170809_00138_TA']
    one_hot = one_hot_all[-3,:]
    # for iter in range(one_hot_all.shape[-1]):
    condition_subject = test_subjects_list[iter]
    # one_hot = one_hot_all[iter,:]

    audio = torch.FloatTensor(audio).unsqueeze(0).to(dev)
    template = torch.FloatTensor(template).unsqueeze(0).flatten(-2).to(dev)
    one_hot = torch.FloatTensor(one_hot).unsqueeze(0).to(dev)

    prediction = model.predict(audio, template, one_hot)
    prediction = prediction.squeeze() # (seq_len, V*3)
    np.save(os.path.join(result_path, wav[:-4] + "_condition_" + condition_subject + ".npy"), prediction.detach().cpu().numpy())



def main():
    parser = argparse.ArgumentParser(description='TalkingStyle: A Novel Approach for the Generation of Personalized and Stylized Talking Face Avatars')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "/workspace/CodeTalker_first_test/BIWI/wav_all", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="/workspace/CodeTalker_first_test/BIWI/vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save/models", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="/workspace/TalkingStyle/styletalk/result", help='path to the predictions')
    # parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    # parser.add_argument("--val_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    # parser.add_argument("--test_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_wav_path", type=str, default='/workspace/TalkingStyle/BIWI/wav')
    parser.add_argument("--lip_region", type=str, default="BIWI/regions/lip.txt", help='path to the lip region')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    args = parser.parse_args()

    # build model
    model = TalkingStyle(args)

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(args.device)
    
    demo_audio = args.test_wav_path

    for file in os.listdir(demo_audio):
        test(args, model, file, args.test_wav_path, epoch=90)


if __name__ == '__main__':
    main()