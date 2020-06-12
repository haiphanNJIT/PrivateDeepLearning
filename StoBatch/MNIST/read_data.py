import pyperclip
import xlwt
import numpy as np

def read_max():
    # read data
    all_lines = []
    for i in range(4):
        with open('TDPAT_max_{}.txt'.format(i+1), 'r') as inf:
            all_lines.append(inf.readlines())

    n_line = 196
    n_d = 10
    steps_list = []
    epss_list = []
    benigns_list = []
    ifgsms_list = []
    fgsms_list = []
    mims_list = []
    madrys_list = []
    for lines in all_lines:
        benigns = []
        ifgsms = []
        fgsms = []
        mims = []
        madrys = []
        for j in range(n_d):
            d_line = lines[j]
            data = d_line.split()
            benigns.append(float(data[1]))
            ifgsms.append(float(data[5]))
            fgsms.append(float(data[3]))
            mims.append(float(data[7]))
            madrys.append(float(data[9]))
        benigns_list.append(benigns)
        ifgsms_list.append(ifgsms)
        fgsms_list.append(fgsms)
        mims_list.append(mims)
        madrys_list.append(madrys)
    print(len(benigns_list))
    print(len(benigns_list[0]))
    #print(len(benigns_list))
    avg_benign = np.mean(np.asarray(benigns_list), axis=0)
    avg_ifgsm = np.mean(np.asarray(ifgsms_list), axis=0)
    avg_fgsm = np.mean(np.asarray(fgsms_list), axis=0)
    avg_mim = np.mean(np.asarray(mims_list), axis=0)
    avg_madry = np.mean(np.asarray(madrys_list), axis=0)

    print(avg_benign)
    print(avg_ifgsm)
    print(avg_fgsm)
    print(avg_mim)
    print(avg_madry)

    all_eps = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
    # for x in step:
    #     print(x)
    # print('==========')
    # a = input('next')
    print('Eps')
    s = 'Eps\n' + '\n'.join([str(x) for x in all_eps])
    pyperclip.copy(s)
    for x in all_eps:
        print(x)
    print('==========')
    a = input('eps print')

    print('Benign')
    s = 'Benign\n' + '\n'.join([str(x) for x in avg_benign])
    pyperclip.copy(s)
    for x in avg_benign:
        print(x)
    print('==========')
    a = input('avg_benign print')

    print('IFGSM')
    s = 'IFGSM\n' + '\n'.join([str(x) for x in avg_ifgsm])
    pyperclip.copy(s)
    for x in avg_ifgsm:
        print(x)
    print('==========')
    a = input('avg_ifgsm print')

    print('FGSM')
    s = 'FGSM\n' + '\n'.join([str(x) for x in avg_fgsm])
    pyperclip.copy(s)
    for x in avg_fgsm:
        print(x)
    print('==========')
    a = input('avg_fgsm print')

    print('Mim')
    s = 'Mim\n' + '\n'.join([str(x) for x in avg_mim])
    pyperclip.copy(s)
    for x in avg_mim:
        print(x)
    print('==========')
    a = input('avg_mim print')

    print('Madry')
    s = 'Madry\n' + '\n'.join([str(x) for x in avg_madry])
    pyperclip.copy(s)
    for x in avg_madry:
        print(x)
    print('avg_madry print')

def find_max():
    n_line = 180
    n_pre = 13
    n_pro = 2

    # read data
    data_file_n = 4
    # with open('DPATResults_v5_IFGSM_many_attacks_run_{}.txt'.format(data_file_n), 'r') as inf:
    with open('TDPATResults_many_attacks_run_{}.txt'.format(data_file_n), 'r') as inf:
        all_lines = inf.readlines()

    n_d = 0
    benign_offset = 5
    ifgsm_offset = 7
    fgsm_offset = 9
    mim_offset = 13
    madry_offset = 11

    # benign_offset = 5
    # ifgsm_offset = 15
    # fgsm_offset = 9
    # mim_offset = 11
    # madry_offset = 13
    s = ''
    
    for n_d in range(10):
        begin = n_d*(n_line+n_pre+n_pro+1)+n_pre
        end = begin + 181
        d_lines = all_lines[begin:end]
        
        #data = ''.join(d_lines).split()
        #print(data)

        step = []
        eps = []
        benign = []
        ifgsm = []
        fgsm = []
        mim = []
        madry = []
        for line in d_lines:
            data = line.split()
            try:
                benign.append(float(data[benign_offset]))
                ifgsm.append(float(data[ifgsm_offset]))
                fgsm.append(float(data[fgsm_offset]))
                mim.append(float(data[mim_offset]))
                madry.append(float(data[madry_offset]))
            except:
                print(line)
        max_benign = max(benign)
        max_ifgsm = max(ifgsm)
        max_fgsm = max(fgsm)
        max_mim = max(mim)
        max_madry = max(madry)

        s += 'max_benign: {} max_fgsm: {} max_ifgsm: {} max_mim: {} max_madry: {}\n'.format(max_benign, max_fgsm, max_ifgsm, max_mim, max_madry)
    
        print('max_benign: {} max_fgsm: {} max_ifgsm: {} max_mim: {} max_madry: {}'.format(max_benign, max_fgsm, max_ifgsm, max_mim, max_madry))
    # pyperclip.copy('max_benign: {} max_fgsm: {} max_ifgsm: {} max_mim: {} max_madry: {}'.format(max_benign, max_fgsm, max_ifgsm, max_mim, max_madry))
    pyperclip.copy(s)

#find_max()
read_max()
    # print(len(steps_list))
    # print(len(steps_list[0]))
    # print(len(steps_list[0][0]))
    # exit()
    # #[4, 10, 180]
    # avg_benign = np.mean(np.asarray(benigns_list), axis=0)
    # avg_ifgsm = np.mean(np.asarray(ifgsms_list), axis=0)
    # avg_fgsm = np.mean(np.asarray(fgsms_list), axis=0)
    # avg_mim = np.mean(np.asarray(mims_list), axis=0)
    # avg_madry = np.mean(np.asarray(madrys_list), axis=0)

#step = steps_list[0][0]
# all_eps = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]

# wb = xlwt.Workbook()
# ws = wb.add_sheet('Sheet1') 
# # (row, col)
# row_title = ['step', 'benign', 'ifgsm', 'fgsm', 'mim', 'madry']

# for col_base in range(n_d):
#     ws.write(0, 7*col_base, 'DPAT_{}'.format(all_eps[col_base]))
#     for i in range(len(row_title)):
#         ws.write(1, 7*col_base+i, row_title[i])
#     for row_base in range(n_line):
#         ws.write(2+row_base, 7*col_base, step[row_base])
#         ws.write(2+row_base, 7*col_base+1, avg_benign[row_base])
#         ws.write(2+row_base, 7*col_base+2, avg_ifgsm[row_base])
#         ws.write(2+row_base, 7*col_base+3, avg_fgsm[row_base])
#         ws.write(2+row_base, 7*col_base+4, avg_mim[row_base])
#         ws.write(2+row_base, 7*col_base+5, avg_madry[row_base])

# wb.save('test1.xls')
# for x in step:
#     print(x)
# print('==========')
# a = input('next')
# for x in benign:
#     print(x)
# print('==========')
# a = input('next')
# for x in ifgsm:
#     print(x)
# print('==========')
# a = input('next')
# for x in fgsm:
#     print(x)
# print('==========')
# a = input('next')
# for x in mim:
#     print(x)
# print('==========')
# a = input('next')
# for x in madry:
#     print(x)
# #print(s.split())