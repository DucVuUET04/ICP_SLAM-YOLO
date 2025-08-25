import shutil,os

# remove all trong folder
def remove_all(path):
    if os.path.exists(path) == True:
        l = 0
        for i in list(path):
            if i == ".":
                l = 1
                try:
                    os.remove(path) 
                except OSError as e:
                    print('Không xóa được file/folder: ' + path)
                break
        if l == 0:
            try:
                shutil.rmtree(path)
            except OSError as e:
                print('Không xóa được file/folder: ' + path)
    else:
        print('Không tồn tại file/folder: ' + path)

#remove file trong folder
def remove_file(path):
    if os.path.exists(path) == True:
        try:
            os.remove(path) 
        except OSError as e:
            print('Không xóa được file: ' + path)
#remove folder
def remove_folder(path):
    if os.path.exists(path) == True:
        try:
            shutil.rmtree(path)
        except OSError as e:
            print('Không xóa được folder: ' + path)
# remove file in folder
def remove_all_file_in_folder(path,path_no_remove = ""):
    if os.path.exists(path) == True:
        ds = os.listdir(path)
        for i in range(0,len(ds)):
            for i2 in list(ds[i]):
                if i2 == ".":
                    if path+"/"+ds[i] != path_no_remove:
                        remove_all(path+"/"+ds[i])
                    break
#tao folder
def tao_folder(path_folder):
    if os.path.exists(path_folder) == False:
        try:
            # os.mkdir(path_folder)
            os.makedirs(path_folder)
        except OSError as e:
            print("không tạo được folder: " + path_folder)
    else:
        # print("Folder đã tồn tại: " + path)
        pass
    return path_folder
