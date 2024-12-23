import os
import argparse

def get_wavlist(path):
    wav_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                wav_list.append(os.path.join(root, file))
    return wav_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to the directory containing wav files')
    parser.add_argument('--output', type=str, required=False, help='Path to the output file')
    args = parser.parse_args()
    wav_list = get_wavlist(args.path)
    for wav in wav_list:
        print(wav)

    if args.output:
        with open(args.output, 'w') as f:
            for wav in wav_list:
                f.write(wav + '\n')
        print('Wav list saved to', args.output)