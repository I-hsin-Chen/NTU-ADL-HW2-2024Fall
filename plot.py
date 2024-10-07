import json
import matplotlib.pyplot as plt


def plot_rouge_scores(file_path, output_path):

    with open(file_path, 'r') as f:
        data = json.load(f)

    epochs = [entry['epoch'] for entry in data]
    rouge1_scores = [entry['rouge1'] for entry in data]
    rouge2_scores = [entry['rouge2'] for entry in data]
    rougeL_scores = [entry['rougeL'] for entry in data]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rouge1_scores, label='ROUGE-1', color='blue', marker='o')
    plt.plot(epochs, rouge2_scores, label='ROUGE-2', color='green', marker='o')
    plt.plot(epochs, rougeL_scores, label='ROUGE-L', color='red', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('ROUGE Score')
    plt.title('ROUGE Scores over Epochs')
    plt.legend()

    plt.grid(True)
    plt.savefig(output_path)

file_path = 'output/v2/result.json'
output_path = 'output/v2/curve.png'   
plot_rouge_scores(file_path, output_path)
