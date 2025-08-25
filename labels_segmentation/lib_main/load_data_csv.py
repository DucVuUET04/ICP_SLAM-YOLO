import path
from lib_main import edit_csv_tab
import shutil
from tkinter.messagebox import showerror, showwarning, showinfo
import os

path_phan_mem = path.path_phan_mem
path_admin = path_phan_mem + "/setting/admin.csv"

def edit_path(input):
    output = ""
    for i in input:
        if i == "\\":
            output = output + "/"
        else:
            output = output + i
    return output
def load_file_csv(path_input,so_dong_min = 2):
    ds_input = []
    ten_input = []
    tt_input = []
    if os.path.exists(path_input) == False:
        showerror(title='Error',message='Thiếu file setting: ' + str(path_input))
    else:
        ds_input = edit_csv_tab.load_all_stt(path_input)
        ten_input = []
        tt_input = []
        if len(ds_input) >= so_dong_min:
            for i in range(0,len(ds_input)):
                if len(ds_input[i]) >= 1:
                    ten_input.append(ds_input[i][0])
                else:
                    ten_input.append("")
                if len(ds_input[i]) >= 2:
                    tt_input.append(ds_input[i][1:])
                else:
                    tt_input.append([""])
        else:
            showerror(title='Error',message='Kiểm tra lại file: ' + str(path_input))
        return ds_input,ten_input,tt_input

def load_log_csv(path_input,so_dong_min = 1):
    ds_input = []
    ten_input = []
    tt_input = []
    if os.path.exists(path_input) == False:
        showerror(title='Error',message='Thiếu file setting: ' + str(path_input))
    else:
        ds_input = edit_csv_tab.load_all(path_input)
        ten_input = []
        tt_input = []
        if len(ds_input) >= so_dong_min:
            for i in range(0,len(ds_input)):
                if len(ds_input[i]) >= 1:
                    ten_input.append(ds_input[i][0])
                else:
                    ten_input.append("")
                if len(ds_input[i]) >= 1:
                    tt_input.append(ds_input[i][0:])
                else:
                    tt_input.append([""])
        else:
            showerror(title='Error',message='Kiểm tra lại file: ' + str(path_input))
        return ds_input
    
# load danh sach admin
def ds_admin():
    ds,ten,tt = load_file_csv(path_admin)
    return ds,ten,tt

# # print(ds_combobox())

# # load data setting thanh cong xu
# def setting_cong_cu():
#     ds_setting_cc,ten_setting_cc,tt_setting_cc = load_file_csv(path.path_setting_thanh_cc)
#     out_put = []
#     for i in ds_setting_cc:
#         out_put.append(i[1:])
#     return out_put
# # print(setting_cong_cu())
# def load_link_pdf(path):
#     _,_,ds_link_pdf = load_file_csv(path,1)
#     list_link_1 = []
#     list_link_2 = []
#     list_link_3 = []
#     for i in range(0,len(ds_link_pdf)):
#         if ds_link_pdf[i][0] == "":
#             break
#         list_link_1.append(str(i+1)+"_"+str(ds_link_pdf[i][0])+"_canvaslink")
#         list_link_2.append(edit_path((ds_link_pdf[i][0])))
#     return list_link_1,list_link_2

# print(load_link_pdf("D:/Check_carton_PSU/software/setting/danh_sach_duong_link_pdf.csv")[1])

# load danh sach setting
def ds_data(path_data):
    ds_data,ten_data,tt_data = load_file_csv(path_data)
    return ds_data,ten_data,tt_data
