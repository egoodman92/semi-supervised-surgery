from sklearn.metrics import average_precision_score, precision_score, accuracy_score, recall_score
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataset import SurgeryDataset
import torch
import cv2
import random
from torch.nn.functional import softmax


def create_video_clip(img_array, fps, output_path='clips/video.avi'):
    img = img_array[0]
    size = (img.shape[0], img.shape[1])
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def write_accuracy(writer, mode, data_loader, y_pred, y_true, y_score, epoch):
    category_accuracies = {}

    accuracy = accuracy_score(y_true, y_pred)
    writer.add_scalar('Accuracy/' + mode, accuracy, epoch)

    for i, category in enumerate(data_loader.dataset.categories):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        precision = precision_score(y_true[:, i], y_pred[:, i])
        avg_precision = average_precision_score(y_true[:, i], y_score[:, i])
        recall = recall_score(y_true[:, i], y_pred[:, i])
        category_accuracies[category] = acc
        writer.add_scalar('Accuracy/' + mode + "-" + category, acc, epoch)
        writer.add_scalar('Precision/' + mode + "-" + category, precision, epoch)
        writer.add_scalar('Avg_Precision/' + mode + "-" + category, avg_precision, epoch)
        writer.add_scalar('Recall/' + mode + "-" + category, recall, epoch)
        print("%s %s precision: %.4f" % (category, mode, precision))
        print("%s %s average precision: %.4f" % (category, mode, avg_precision))
        print("%s %s recall: %.4f" % (category, mode, recall))
        print("%s %s accuracy: %.4f" % (category, mode, acc))

    print("%s accuracy: %.4f \n" % (mode,  accuracy))
    if mode == 'val' and accuracy > 1.0 / len(data_loader.dataset.categories):
        print("\n\n%.4f accuracy generalizes better than chance \n\n" % accuracy)


def write_video(writer, net, dataset, y_score, y_true, samples=10):
    random_samples = random.sample(range(dataset.__len__()), samples)
    for i in random_samples:
        (data, record_id, _) = dataset.__getitem__(i)
        data = data.unsqueeze(0)
        output, _ = net(data)
        num_batches = data.shape[0]
        output = output.view(-1, int(output.shape[0]/num_batches), output.shape[1])
        output = output.mean(1)
        output = softmax(output)
        prediction = output[0]


        (raw_frames, labels), record = dataset.frames_labels_meta(i)
        input = SurgeryDataset.raw_frames_to_input(raw_frames, num_segments=1)
        input = input.transpose([0, 1, 4, 2, 3])
        input = torch.from_numpy(input)
        label_index = np.argmax(y_true[i])

        score = prediction[label_index]
        label = SurgeryDataset.categories[label_index]

        if 'video_name' in record:
            segment_name = record['video_name'] + ("?start=%d&end=%d" %
                                                 (record['start_seconds'], record['end_seconds']))
        else:
            segment_name = record['filename'].replace(".pkl", "")
            segment_name.replace("train/", "").replace("val/", "")

        video_name = "-".join([segment_name, label, ("%.2f" % score)])
        video_name = "val/" + video_name
        writer.add_video(video_name, input, 1, fps=2)

