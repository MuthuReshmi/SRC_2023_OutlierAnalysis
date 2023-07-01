from flask import *
import pandas as pd
import os
import csv
from csv import DictWriter
from csv import writer
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import zipfile
from zipfile import ZipFile
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)
# individual parameter calculation
# parameter 1
def p2p(df, vinn):
    vinn_max = df[vinn].max()
    vinn_min = df[vinn].min()
    avpp = 0  # added
    avpp = vinn_max - vinn_min
    return avpp
# parameter 2
def meancalc(df, vinn):
    temp = df[vinn].mean()
    return temp
# parameter 3
def stdcalc(df, vinn):
    temp = df[vinn].std()
    return temp
# parameter 4
def rmscalc(df, vinn):
    x = df[vinn] * df[vinn]
    val = (x.mean()) ** (1 / 2)
    return val
# parameter 5
def mavcalc(df, vinn):
    x = abs(df[vinn])
    temp = x.mean()
    return temp
# parameter 6
def energycalc(df, vinn):
    temp = abs(df[vinn]) * abs(df[vinn])
    x = temp.sum()
    return x
# parameter 7
def log_energycalc(df, vinn):
    temp = (df[vinn]) * (df[vinn])
    x = np.log10(temp)
    x = pd.Series.to_numpy(x)
    a = len(x)
    z = np.zeros(a)
    for i in range(a):
        z[i] = '%.2f' % x[i]
    y = 0.0
    for i in range(1, a):
        y = y + z[i]
    return y
# parameter 8
def shaentropy_calc(df, vinn):
    temp = (df[vinn]) * (df[vinn])
    x = np.log10(temp)
    a = x + temp
    a = pd.Series.to_numpy(a)
    b = len(a)
    z = np.zeros(b)
    for i in range(b):
        z[i] = '%.2f' % a[i]
    y = 0.0
    for i in range(1, b):
        y = -(y + z[i])
    return y
@app.route('/',methods=['GET','POST'])  
def main():  
    return render_template("input_html.html")  
@app.route('/app1_fileupload',methods=['GET','POST'])

def app1_fileupload():
   if request.method == "POST":
       f = request.files['file[]']
       f.save(f.filename)       
       features_list=request.form.getlist('Features')
       print(features_list)
       dir_path=os.path.dirname(__file__)
       new_extract_path=dir_path+"\\test\\"
       
    #    print(os.path.join(ROOT_DIR, 'data', 'mydata.json'))
# loading the temp.zip and creating a zip object
       with ZipFile(f.filename, 'r') as zObject:

	# Extracting all the members of the zip
	# into a specific location.
         zObject.extractall(
		 path=new_extract_path)
    # dirs=directories
       name =''
       filelist=[]
       for root, dirs, files in os.walk(new_extract_path):
         for file in files:
            filelist.append(os.path.join(root,file))
         p2pvalues=[]
       meancalcvalues=[] 
       stdcalcvalues=[]
       rmscalcvalues=[]
       mavcalcvalues=[]
       energycalcvalues=[]
       logenergycalcvalues=[]
       shaentropy_calcvalues=[]
       names=[]
       for fi in filelist:                
                name = fi        
                names.append(name)               
                df = pd.read_csv(fi)
                p2pvalue = p2p(df,"vinn")     
                p2pvalues.append(p2pvalue)
                meancalcvalue = meancalc(df,"vinn")
                meancalcvalues.append(meancalcvalue)       
                stdcalcvalue = stdcalc(df,"vinn")
                stdcalcvalues.append(stdcalcvalue)
                rmscalcvalue = rmscalc (df,"vinn")
                rmscalcvalues.append(rmscalcvalue)
                mavcalcvalue = mavcalc(df,"vinn")
                mavcalcvalues.append(mavcalcvalue)
                energycalcvalue = energycalc(df,"vinn")
                energycalcvalues.append(energycalcvalue)
                logenergycalcvalue = log_energycalc(df,"vinn")
                logenergycalcvalues.append(logenergycalcvalue)
                shaentropy_calcvalue = shaentropy_calc(df,"vinn")
                shaentropy_calcvalues.append(shaentropy_calcvalue)          
          #Writing the calculated files to the c
       csv_file_path=dir_path+"\\uploads\\calculated_values_uploaded.csv"
       field_names = ['NAME', 'P2P', 'MEAN', 'MAV', 'STD', 'RMS','ENERGY', 'LOG_ENERGY','SHA_ENTROPY']
       features = {'NAME':names, 'P2P':p2pvalues, 'MEAN':meancalcvalues, 'MAV':mavcalcvalues, 'STD':stdcalcvalues, 'RMS':rmscalcvalues,'ENERGY':energycalcvalues, 'LOG_ENERGY':logenergycalcvalues,'SHA_ENTROPY':shaentropy_calcvalues}
    #writer() function returns a writer object that converts the user's data into a delimited string. This string can later be used to write into CSV files using the writerow() function
       with open(csv_file_path, 'w') as f_object:     
            writer_object = writer(f_object)
            writer_object.writerow(field_names)
            writer_object.writerows(zip(*features.values()))
            f_object.close()
    #Converting the same file to a dataframe
       df=pd.read_csv(csv_file_path)
       image_path=dir_path+"\\static\\"
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['P2P'].plot(kind='box')
       title = "P2P"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"P2P.jpg")
       
       
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['MEAN'].plot(kind='box')
       title = "MEAN"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"mean.jpg")
    
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['STD'].plot(kind='box')
       title = "STD"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"std.jpg")
    
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['RMS'].plot(kind='box')
       title = "RMS"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"rms.jpg")
    
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['MAV'].plot(kind='box')
       title = "MAV"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"mav.jpg")
    
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['ENERGY'].plot(kind='box')
       title = "ENERGY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"energy.jpg")
    
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['LOG_ENERGY'].plot(kind='box')
       title = "LOG_ENERGY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"log energy.jpg")

    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['SHA_ENTROPY'].plot(kind='box')
       title = "SHA_ENTROPY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"sha entropy.jpg")
       images=[]
       for features in features_list:
           image_name=features+".jpg"
           images.append(image_name)
       print(images)
       return render_template('box_plots_display.html', images=images)       
if __name__ == '__main__':
    app.run(debug=True)