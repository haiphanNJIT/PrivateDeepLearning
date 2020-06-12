import numpy as np
import matplotlib.pyplot as plt
import os, sys

result_list = ["mnist_mlp_pre_train_for_eval_test.log","mnist_mlp_adv_pre_train_for_eval_test.log",
    "mnist_mlp_noise_pre_train_for_eval_test.log","mnist_mlp_noise_adv_pre_train_for_eval_test.log"]
result_dict = {}
available_sources = ["mnist_mlp_pre_train_for_bbox","mnist_conv_A_pre_train_for_bbox","mnist_conv_B_pre_train_for_bbox","mnist_conv_C_pre_train_for_bbox"]

def read_result(path):
    benign_err = 0.0
    wbox_err_dict = {}
    bbox_err_dict = {}
    with open(path,'r') as inf:
        line = inf.readline()
        benign_err = float(line.split()[-1])
        for i in range(6):
            words = inf.readline().split()
            print(words)
            wbox_err_dict[words[1]] = float(words[-1])
        for i in range(4):
            blank = inf.readline()
            words = inf.readline().split()
            print(words)
            source = words[-1]
            bbox_err_dict[source] = {}
            for j in range(6):
                words = inf.readline().split()
                bbox_err_dict[source][words[1]] = float(words[-1])
    return benign_err, wbox_err_dict, bbox_err_dict

def plot_wbox_result(path):
    fig, ax = plt.subplots()
    ind = np.arange(7)
    legend_list = []
    width = 0.2
    current_width = 0.0
    color = ['red','blue','orange','green']
    count = 0
    for test in result_list:
        p = os.path.join(path, test)
        benign_err, wbox_err_dict, bbox_err_dict = read_result(p)
        test_results = [benign_err,
            wbox_err_dict["fgsm"],wbox_err_dict["targeted_fgsm"],
            wbox_err_dict["ifgsm"],wbox_err_dict["targeted_ifgsm"],
            wbox_err_dict["randfgsm"],wbox_err_dict["targeted_randfgsm"]]
        series = ax.bar(ind+current_width, test_results, width, color = color[count])
        current_width += width
        legend_list.append(series)
        count += 1
    ax.set_ylabel('Erro rates')
    ax.set_title('Error rates, white box attacks on MLP model, MNIST')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('benign','fgsm','t_fgsm','ifgsm','t_ifgsm','randfgsm','t_randfgsm'))
    ax.legend(tuple(x[0] for x in legend_list), ('Normal training', 'Ensemble Adv training', 'Noise training', 'Noise + adv training'))
    plt.show()

def plot_bbox_result(path, source):
    fig, ax = plt.subplots()
    ind = np.arange(7)
    legend_list = []
    width = 0.2
    current_width = 0.0
    color = ['red','blue','orange','green']
    count = 0
    for test in result_list:
        p = os.path.join(path, test)
        benign_err, wbox_err_dict, bbox_err_dict = read_result(p)
        test_results = [benign_err,
            bbox_err_dict[source]["fgsm"],bbox_err_dict[source]["targeted_fgsm"],
            bbox_err_dict[source]["ifgsm"],bbox_err_dict[source]["targeted_ifgsm"],
            bbox_err_dict[source]["randfgsm"],bbox_err_dict[source]["targeted_randfgsm"]]
        series = ax.bar(ind+current_width, test_results, width, color = color[count])
        current_width += width
        legend_list.append(series)
        count += 1
    ax.set_ylabel('Erro rates')
    ax.set_title('Error rates, black box attacks sourced from {} on MLP model, MNIST'.format(source))
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('benign','fgsm','t_fgsm','ifgsm','t_ifgsm','randfgsm','t_randfgsm'))
    ax.legend(tuple(x[0] for x in legend_list), ('Normal training', 'Ensemble Adv training', 'Noise training', 'Noise + adv training'))
    plt.show()


p = os.path.join(os.getcwd(), 'experiment', 'test_pre_trained_models')
#p = os.path.join(os.getcwd(), 'experiment', 'large_noise_inout_pre_trained_models')
#p = os.path.join(os.getcwd(), 'experiment', 'mid_noise_inout_pre_trained_models')
#p = os.path.join(os.getcwd(), 'experiment', 'small_noise_inout_pre_trained_models')
plot_wbox_result(p)
#plot_bbox_result(p,"mnist_mlp_pre_train_for_bbox")
#plot_bbox_result(p,"mnist_conv_A_pre_train_for_bbox")
#plot_bbox_result(p,"mnist_conv_B_pre_train_for_bbox")
#plot_bbox_result(p,"mnist_conv_C_pre_train_for_bbox")
