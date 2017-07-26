"""
Given a training log file, plot something.
"""
import csv
import matplotlib.pyplot as plt

def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        accuracies = []
        top_5_accuracies = []
        cnn_benchmark = []  # this is ridiculous
        for epoch,acc,loss,val_acc,val_loss, in reader:
            accuracies.append(float(val_acc))
            top_5_accuracies.append(float(acc))
            cnn_benchmark.append(0.65)  # ridiculous

        plt.plot(accuracies, label = "Validation Accuracy")
        plt.plot(top_5_accuracies ,label = "Training Accuracy")
        plt.plot(cnn_benchmark, label= "Predefine Benchmark")
        plt.legend(loc = "bottom right")
        plt.show()

if __name__ == '__main__':
    training_log = 'data/logs/-training-1500997293.14.log'
    main(training_log)
