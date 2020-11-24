import os
import string
import re

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub('fig[0-9]+',' ',text)
    # text = re.sub('[^a-zA-Z0-9]',' ',text) 
    text = re.sub('\w{1,}[0-9]+', ' ', text)
    text = re.sub('[0-9]', '', text)
    text = re.sub(r"[-()\"#/@;:<>{}`_\\+=~|!?,]", "", text)
    text = re.sub('^a$','', text)
    return text

mypath = 'bigPatentData/train/a'
orig = os.listdir(mypath)
ctr = 0
mydest = 'final_train'
for filename in orig:
    f2 = open(mypath+'/'+filename, mode = 'r')
    dataset = f2.read()
    dataset = dataset.split('}')
    x = []
    y = []
    for itm in dataset:
        abst =  itm.split("\"abstract\":",1)
        if len(abst)!=1:
            abst = abst[1].split("\"application_number\":")[0]
            y.append(abst)
        full_txt = itm.split("\"description\":")
        if len(full_txt)>1:
            full_txt = full_txt[1]
            x.append(full_txt[1])
            # print("full text:"+ full_txt)
            # print("abstract:" + abst)
            f = open(mydest + "/" + "train"+ str(ctr) +".story", "a")
            full_txt = clean_text(full_txt)
            if ctr == 0:
                print(full_txt)
            full_txt = full_txt.split('.')
            sep = '.\n\n'
            full_txt = sep.join(full_txt)
            f.write(full_txt)
            sep = '\n\n' + '@highlight' + '\n\n'
            f.write(sep)
            abst = clean_text(abst)
            if len(abst.split('.'))>1:
                abst = abst.split('.')
                abst = sep.join(abst)
            f.write(abst)
            f.close()
            ctr = ctr + 1

        
    del dataset
            




