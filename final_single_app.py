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
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pickle
import zipfile
from zipfile import ZipFile
import warnings
import shutil
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
app = Flask(__name__)
dir_path=os.path.dirname(__file__)
def create_static_folder():
    folder_path = dir_path+"//static"   
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print("Static folder created successfully.")
    else:
        print("Static folder already exists.")
# Call the function to create the "static" folder if it doesn't exist
create_static_folder()
def create_uploads_folder():
    folder_path = dir_path+"//uploads"
    
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print("Uploads folder created successfully.")
    else:
        print("Uploads folder already exists.")

# Call the function to create the "static" folder if it doesn't exist
create_uploads_folder()
def create_test_folder():
    folder_path = dir_path+"//test"
    
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print("Test folder created successfully.")
    else:
        print("Test folder already exists.")

# Call the function to create the "static" folder if it doesn't exist
create_test_folder()
# def delete_folders(folder_path):
#     # Iterate over all the items in the given folder
#     for item in os.listdir(folder_path):
#         item_path = os.path.join(folder_path, item)
        
#         # Check if the item is a directory
#         if os.path.isdir(item_path):
#             # Recursively delete the directory and its contents
#             delete_folders(item_path)
            
#             # Delete the empty directory
#             os.rmdir(item_path)

# # Specify the path of the folder containing the folders you want to delete
# folder_to_delete = dir_path+"//test"

# # Call the function to delete all folders within the specified folder
# delete_folders(folder_to_delete)
csv_file_path=dir_path+"//uploads//calculated_values_uploaded_train.csv"
csv_file_path1=dir_path+"//uploads//opamp_parameter_scaled.csv"
csv_file_path2=dir_path+"//uploads//kmeans_opamp.csv"
csv_file_path3=dir_path+"//uploads//svm_opamp.csv"
csv_file_path4=dir_path+"//uploads//dbscan_opamp.csv"
csv_file_path5=dir_path+"//uploads//opamp_pca.csv"
csv_file_path7=dir_path+"//uploads//opamp_test_pca.csv"
csv_file_path6=dir_path+"//uploads//opamp_pca_clustering.csv"
csv_file_path8=dir_path+"//uploads//opamp_pca_test_clustering.csv"
csv_file_path9=dir_path+"//uploads//calculated_values_uploaded_test.csv"
csv_file_path11=dir_path+"//uploads//calculated_values_uploaded_train_ac.csv"
csv_file_path12 = dir_path+"//uploads//ac_parameter_scaled.csv"
csv_file_path13=dir_path+"//uploads//kmeans_ac.csv"
csv_file_path14=dir_path+"//uploads//svm_ac.csv"
csv_file_path15=dir_path+"//uploads//ac_pca.csv"
csv_file_path16=dir_path+"//uploads//ac_pca_clustering.csv"
csv_file_path17=dir_path+"//uploads//ac_test_pca.csv"
csv_file_path18=dir_path+"//uploads//ac_pca_test_clustering.csv"
csv_file_path19=dir_path+"//uploads//calculated_values_uploaded_test_ac.csv"
csv_file_path20=dir_path+"//uploads//dbscan_ac.csv"
pickle_path_kmeans=dir_path+"//kmodel_opamp.pkl"
pickle_path_kmeans_ac=dir_path+"//kmodel_ac.pkl"
pickle_path_svm=dir_path+"//smodel_opamp.pkl"
pickle_path_svm_ac=dir_path+"//smodel_ac.pkl"
pickle_path_dbscan=dir_path+"//dmodel_opamp.pkl"
pickle_path_svm_pca=dir_path+"//smodel_opamp_pca.pkl"
pickle_path_dbscan_pca=dir_path+"//dmodel_opamp_pca.pkl"
pickle_path_dbscan_ac=dir_path+"//dmodel_ac.pkl"
pickle_path_svm_pca_ac=dir_path+"//smodel_ac_pca.pkl"
pickle_path_dbscan_pca_ac=dir_path+"//dmodel_ac_pca.pkl"
csv_file_path0=dir_path+"//uploads//calculated_values_uploaded_train.csv"
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
def find_outliers_IQR(df):
    q1 = df.iloc[:, 0].quantile(0.25)
    q3 = df.iloc[:, 0].quantile(0.75)
    IQR = q3 - q1
    outliers = df[((df.iloc[:, 0] < (q1 - 1.5 * IQR)) | (df.iloc[:, 0] > (q3 + 1.5 * IQR)))]
    return outliers
def find_outliers_IQR_AC(df):
    q1 = df.iloc[:, 1].quantile(0.25)
    q3 = df.iloc[:, 1].quantile(0.75)
    IQR = q3 - q1
    outliers = df[((df.iloc[:, 1] < (q1 - 1.5 * IQR)) | (df.iloc[:, 1] > (q3 + 1.5 * IQR)))]
    return outliers
@app.route('/',methods=['GET','POST'])  
def main():  
    return render_template("Main_Home_Page.html")  
@app.route('/home_page_selection',methods=['GET','POST'])  
def home_page_selection():  
    if request.method == "POST":
        value = request.form.get('Circuit')
    if value=='1':
        return redirect(url_for('home_page'))
    elif value=='2':
        return redirect(url_for('home_page_ac'))    
@app.route('/home_page',methods=['GET','POST'])  
def home_page():  
    return render_template("input_html.html")  
@app.route('/app1_fileupload',methods=['GET','POST'])
def app1_fileupload():
   if request.method == "POST":
       f = request.files['file[]']
       f.save(f.filename)       
       features_list=request.form.getlist('Features')
       if(features_list=='Select all'):
           features_list=['P2P','Mean','STD','RMS','MAV','Energy','Log Energy','Sha Entropy']
       print(features_list)
       dir_path=os.path.dirname(__file__)
       new_extract_path=dir_path+"//test//"
       with ZipFile(f.filename, 'r') as zObject:
         zObject.extractall(
		 path=new_extract_path)

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
                name=os.path.basename(fi).split('/')[-1]
                names.append(name)               
                df = pd.read_csv(fi)
                if('vinn' in df.columns):
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
                else:
                    return render_template('Wrong_input.html')       
          #Writing the calculated files to the c
       csv_file_path0=dir_path+"//uploads//calculated_values_uploaded_train.csv"
       field_names = ['NAME', 'P2P', 'MEAN', 'MAV', 'STD', 'RMS','ENERGY', 'LOG_ENERGY','SHA_ENTROPY']
       features = {'NAME':names, 'P2P':p2pvalues, 'MEAN':meancalcvalues, 'MAV':mavcalcvalues, 'STD':stdcalcvalues, 'RMS':rmscalcvalues,'ENERGY':energycalcvalues, 'LOG_ENERGY':logenergycalcvalues,'SHA_ENTROPY':shaentropy_calcvalues}
    #writer() function returns a writer object that converts the user's data into a delimited string. This string can later be used to write into CSV files using the writerow() function
       with open(csv_file_path, 'w') as f_object:     
            writer_object = writer(f_object)
            writer_object.writerow(field_names)
            writer_object.writerows(zip(*features.values()))
            f_object.close()
    #Converting the same file to a dataframe
       df=pd.read_csv(csv_file_path0)
       image_path=dir_path+"//static//"
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
       if(features_list=='Select all'):
           features_list=['P2P','Mean','STD','RMS','MAV','Energy','Log Energy','Sha Entropy']
       for features in features_list:
           
           image_name=features+".jpg"    
           images.append(image_name)
       df = pd.read_csv(csv_file_path)      
       stat = df.describe()
       stat.to_csv(dir_path+"//uploads//opamp_stat.csv")
   
       outlier_p2p = find_outliers_IQR(df.iloc[:,[1,0]])
       outlier_p2p.to_csv(dir_path+"//uploads//opamp_outliers_P2P.csv")
    
       outlier_offset = find_outliers_IQR(df.iloc[:,[3,0]])
       outlier_offset.to_csv(dir_path+"//uploads//opamp_outliers_Mav.csv")
 
       outlier_offset = find_outliers_IQR(df.iloc[:,[6,0]])
       outlier_offset.to_csv(dir_path+"//uploads//opamp_outliers_Energy.csv") 
    
       outlier_offset = find_outliers_IQR(df.iloc[:,[7,0]])
       outlier_offset.to_csv(dir_path+"//uploads//opamp_outliers_Log Energy.csv")
    
       outlier_offset = find_outliers_IQR(df.iloc[:,[8,0]])
       outlier_offset.to_csv(dir_path+"//uploads//opamp_outliers_Sha Entropy.csv")
    
       outlier_offset = find_outliers_IQR(df.iloc[:,[5,0]])
       outlier_offset.to_csv(dir_path+"//uploads//opamp_outliers_RMS.csv")

       outlier_offset = find_outliers_IQR(df.iloc[:,[2,0]])
       outlier_offset.to_csv(dir_path+"//uploads//opamp_outliers_Mean.csv")
   
       outlier_offset = find_outliers_IQR(df.iloc[:,[4,0]])
       outlier_offset.to_csv(dir_path+"//uploads//opamp_outliers_STD.csv")
       
       list_df=[]
       if(features_list=='Select all'):
           features_list=['P2P','Mean','STD','RMS','MAV','Energy','Log Energy','Sha Entropy']
       for features in features_list:
           outlier_file_name=dir_path+"//uploads//opamp_outliers_"+features+".csv"
           df=pd.read_csv(outlier_file_name)
           if(df.empty==False):
               df1=df.iloc[:,[1,2]]
               list_df.append(df1)           
       return render_template('box_plots_display.html', images=images,list_df=list_df)   
@app.route('/clustering',methods=['GET','POST'])  
def clustering():  
    return render_template("clustering_main_page.html")  
@app.route('/app2_clustering',methods=['GET','POST'])
def app2_clustering():
    df=pd.read_csv(csv_file_path)
    scaler=MinMaxScaler()
    columns=df.columns.to_list()
    for_scaling=columns[1:10]

    df_fn=df[for_scaling]
    print(df_fn)
    scaled_values=scaler.fit_transform(df_fn)
    scaled_df=pd.DataFrame(scaled_values,columns=[for_scaling])
    scaled_df['Name']=df['NAME']
    csv_file_path1 = dir_path+"//uploads//opamp_parameter_scaled.csv"
    print(scaled_df)
    scaled_df.to_csv(csv_file_path1)
    if request.method == "POST":
        value = request.form.get('Clustering')
    if value=='1':
        return redirect(url_for('app3_KMeansclustering'))
    elif value=='2':
        return redirect(url_for('app4_OneClassSVMClustering'))
    elif value=='3':
        return redirect(url_for('app5_DBScanClustering'))
@app.route('/app3_KMeansclustering',methods=['GET','POST'])
def app3_KMeansclustering():
    return render_template("KMeans_Clustering.html") 
@app.route('/app3_logic',methods=['GET','POST'])
def app3_logic():
    if request.method == "POST":
        num_clusters=request.form.get("Cluster Number")
        num_clusters=int(num_clusters)
        df=pd.read_csv(csv_file_path1)
        X = df.iloc[:,1:9].values
        kmeans = KMeans(num_clusters)
        kmeans.fit(X)
        identified_clusters = kmeans.fit_predict(X)
        df['Clusters'] = identified_clusters 
        #To store the clusters in the file
        pickle.dump(kmeans, open(pickle_path_kmeans,'wb'))
        df.to_csv(csv_file_path2)
    
    return render_template("KMeans_Output.html")
@app.route('/app4_OneClassSVMClustering',methods=['GET','POST'])
def app4_OneClassSVMClustering():
    return render_template('OneClassSVM.html')
@app.route('/app4_logic',methods=['GET','POST'])
def app4_logic():
    if request.method == "POST":
        gamma_num=request.form.get("Gamma")
        gamma_num=float(gamma_num)
        Nu_num=request.form.get("Nu")
        Nu_num=float(Nu_num)
    df=pd.read_csv(csv_file_path1)
    columns=df.columns.to_list()
    columns=columns[1:9]
    df_svm=df[columns].values
    #OneClassSVM using rbf (radial basis function)
    clf=OneClassSVM(kernel='rbf',gamma=gamma_num,nu=Nu_num).fit(df_svm)
    clusters=clf.predict(df_svm)
    for_scaling=columns[0:]
    df=df[for_scaling]
    df['cluster']=clusters
    df.to_csv(csv_file_path3)
    pickle.dump(clf, open(pickle_path_svm,'wb'))
    #PCA Plot
    df=pd.read_csv(csv_file_path)
    sc = StandardScaler()
    scaleddata = sc.fit_transform(df.iloc[:,1:9])
    pca = PCA(n_components = 3)
    dataset_pca = pca.fit_transform(scaleddata)
    df_pca = pd.DataFrame(dataset_pca,columns=['PCA1','PCA2','PCA3'])
    df_pca.to_csv(csv_file_path5)
    data=[]
    for d in df_pca.itertuples():
        my_list =[d.PCA1, d.PCA2, d.PCA3]
        data.extend([my_list])
    clf=OneClassSVM(kernel='rbf',gamma=3,nu=0.001).fit(data)
    clusters=clf.predict(data)
    df_pca['OneClassSVM']=clusters
    pickle.dump(clf, open(pickle_path_svm_pca,'wb'))
    clusters = DBSCAN(eps=1.0, min_samples=15000).fit(data)
    labels=clusters.labels_
    df_pca['DBSCAN'] = clusters.labels_
    pickle.dump(clusters,open(pickle_path_dbscan_pca,'wb'))  
    df_pca.to_csv(csv_file_path6)
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes(projection='3d')
    df_pca_svm=df_pca.iloc[:,0:4]
    df_pca_dscan=df_pca.iloc[:,[col for col in range(len(df_pca.columns)) if col != 3]]
    for i in df_pca_svm:
           xdata = df_pca_svm['PCA1']
           zdata = df_pca_svm['PCA2']
           ydata = df_pca_svm['PCA3']
           ax.scatter3D(xdata, ydata, zdata)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.legend()
    plt.title("3D PCA plot for SVM")
    image_path=dir_path+"//static//"
    plt.savefig(image_path+"PCA_SVM.jpg")
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes(projection='3d')
    for i in df_pca_dscan:
           xdata = df_pca_dscan['PCA1']
           zdata = df_pca_dscan['PCA2']
           ydata = df_pca_dscan['PCA3']
           ax.scatter3D(xdata, ydata, zdata)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.legend()
    plt.title("3D PCA plot for DBSCAN")
    plt.savefig(image_path+"PCA_DBSCAN.jpg")
    return render_template("SVM_Output.html",image="PCA_SVM.jpg")
@app.route('/app5_DBScanClustering',methods=['GET','POST'])
def app5_DBScanClustering():
    return render_template('DBScanClustering.html')
@app.route('/app5_logic',methods=['GET','POST'])
def app5_logic():
    if request.method == "POST":
        epsilon_num=request.form.get("epsi")
        epsilon_num=float(epsilon_num)
    df = pd.read_csv(csv_file_path1)
    df_fn= df.iloc[:,1:9].values
    clusters = DBSCAN(eps=epsilon_num, min_samples=1000).fit(df_fn)
    labels=clusters.labels_
    df['cluster'] = clusters.labels_
    pickle.dump(clusters,open(pickle_path_dbscan,'wb'))
    df.to_csv(csv_file_path4)
    return render_template("DBScan_Output.html",image="PCA_DBSCAN.jpg")
@app.route('/testing_page',methods=['GET','POST'])  
def testing_page():  
    return render_template("input_test_file.html")  
@app.route('/app6_testing',methods=['GET','POST'])
def app6_testing():
   if request.method == "POST":
       f = request.files['file']
       f.save(f.filename)       
       features_list=request.form.getlist('Features')
       if(features_list=='Select all'):
           features_list=['P2P','Mean','STD','RMS','MAV','Energy','Log Energy','Sha Entropy']
       dir_path=os.path.dirname(__file__)
       df = pd.read_csv(f.filename)
       if('vinn' in df.columns):
            name=f.filename
            p2pvalue = p2p(df,"vinn")     
            meancalcvalue = meancalc(df,"vinn")      
            stdcalcvalue = stdcalc(df,"vinn")            
            rmscalcvalue = rmscalc (df,"vinn")            
            mavcalcvalue = mavcalc(df,"vinn")            
            energycalcvalue = energycalc(df,"vinn")            
            logenergycalcvalue = log_energycalc(df,"vinn")            
            shaentropy_calcvalue = shaentropy_calc(df,"vinn")
            
       else:
            return render_template('Wrong_Input.html')          
       field_names = ['NAME', 'P2P', 'MEAN', 'MAV', 'STD', 'RMS','ENERGY', 'LOG_ENERGY','SHA_ENTROPY']
       features = {'NAME':name, 'P2P':p2pvalue, 'MEAN':meancalcvalue, 'MAV':mavcalcvalue, 'STD':stdcalcvalue, 'RMS':rmscalcvalue,'ENERGY':energycalcvalue, 'LOG_ENERGY':logenergycalcvalue,'SHA_ENTROPY':shaentropy_calcvalue}
    #writer() function returns a writer object that converts the user's data into a delimited string. This string can later be used to write into CSV files using the writerow() function
       with open(csv_file_path9, 'w') as f_object:     
            writer_object = writer(f_object)
            writer_object.writerow(field_names)
            writer_object.writerow(features.values())
            f_object.close()
    #Converting the same file to a dataframe
       df=pd.read_csv(csv_file_path9)
       image_path=dir_path+"//static//"
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['P2P'].plot(kind='box')
       title = "P2P"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"P2P_test.jpg")            
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['MEAN'].plot(kind='box')
       title = "MEAN"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"mean_test.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['STD'].plot(kind='box')
       title = "STD"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"std_test.jpg")   
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['RMS'].plot(kind='box')
       title = "RMS"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"rms_test.jpg")  
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['MAV'].plot(kind='box')
       title = "MAV"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"mav_test.jpg")    
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['ENERGY'].plot(kind='box')
       title = "ENERGY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"energy_test.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['LOG_ENERGY'].plot(kind='box')
       title = "LOG_ENERGY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"log energy_test.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['SHA_ENTROPY'].plot(kind='box')
       title = "SHA_ENTROPY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"sha entropy_test.jpg")
       images=[]
       if(features_list=='Select all'):
           features_list=['P2P','Mean','STD','RMS','MAV','Energy','Log Energy','Sha Entropy']
       for features in features_list:
           image_name=features+"_test.jpg"
           images.append(image_name)
           #Converting the same file to a dataframe
       df=pd.read_csv(csv_file_path9)
       data = df.iloc[:,1:10].values
    #Reading the pickle file for the Kmeans
       kmodel = pickle.load(open(pickle_path_kmeans,'rb'))
       k = kmodel.predict(data)
       k = str(k)
    #Reading the pickle file for the SVM
       smodel = pickle.load(open(pickle_path_svm,'rb'))
       s = smodel.predict(data)
       s = str(s)
    #Reading the pickle file for the dbscan
       dmodel=pickle.load(open('dmodel_opamp.pkl','rb'))
       d = dmodel.fit_predict(data, y=None, sample_weight=None)
       d = str(d)           
       return render_template('Cluster_test.html', images=images,k=k,d=d,s=s) 
@app.route('/app7',methods=['GET','POST'])
def app7():
       df=pd.read_csv(csv_file_path0)
       df1=pd.read_csv(csv_file_path9)
       data = df1.iloc[:,0:10].values
       list1=data[0]
       #np.insert(list1,0,"test file")
       df.loc[len(df)] = list1
       sc = StandardScaler()
       scaleddata = sc.fit_transform(df.iloc[:,1:9])
       pca = PCA(n_components = 3)
       dataset_pca = pca.fit_transform(scaleddata)
       df_pca = pd.DataFrame(dataset_pca,columns=['PCA1','PCA2','PCA3'])
       df_pca.to_csv(csv_file_path7)
       data1=[]
       for d in df_pca.itertuples():
        my_list =[d.PCA1, d.PCA2, d.PCA3]
        data1.extend([my_list])
       clf=OneClassSVM(kernel='rbf',gamma=3,nu=0.001).fit(data1)
       clusters=clf.predict(data1)
       df_pca['OneClassSVM']=clusters
       clusters = DBSCAN(eps=1.0, min_samples=15000).fit(data1)
       labels=clusters.labels_
       df_pca['DBSCAN'] = clusters.labels_
       df_pca.to_csv(csv_file_path8)
       fig = plt.figure(figsize=(7,7))
       ax = plt.axes(projection='3d')
       df_pca_svm=df_pca.iloc[:,0:4]
       df_pca_dscan=df_pca.iloc[:,[col for col in range(len(df_pca.columns)) if col != 3]]
       for i in df_pca_svm:
           #spec = training_pca[training_pca['Species']==i]
           xdata = df_pca_svm['PCA1']
           zdata = df_pca_svm['PCA2']
           ydata = df_pca_svm['PCA3']
           ax.scatter3D(xdata, ydata, zdata)
       ax.set_xlabel('PCA1')
       ax.set_ylabel('PCA2')
       ax.set_zlabel('PCA3')
       ax.legend()
       plt.title("3D PCA plot for SVM")
       image_path=dir_path+"//static//"
       plt.savefig(image_path+"PCA_SVM_test.jpg")
       fig = plt.figure(figsize=(7,7))
       ax = plt.axes(projection='3d')
       for i in df_pca_dscan:
           #spec = training_pca[training_pca['Species']==i]
           xdata = df_pca_dscan['PCA1']
           zdata = df_pca_dscan['PCA2']
           ydata = df_pca_dscan['PCA3']
           ax.scatter3D(xdata, ydata, zdata)
       ax.set_xlabel('PCA1')
       ax.set_ylabel('PCA2')
       ax.set_zlabel('PCA3')
       ax.legend()
       plt.title("3D PCA plot for DBSCAN")
       plt.savefig(image_path+"PCA_DBSCAN_test.jpg")
       image1= 'PCA_DBSCAN_Test.jpg'
       image2='PCA_SVM_Test.jpg'
       return render_template('PCATest.html',image1=image1,image2=image2)   
@app.route('/home_page_ac',methods=['GET','POST'])  
def home_page_ac():  
    return render_template("input_html_ac.html")    
@app.route('/app1_fileupload_ac',methods=['GET','POST'])
def app1_fileupload_ac():
   if request.method == "POST":
       f = request.files['file[]']
       f.save(f.filename)       
       features_list=request.form.getlist('Features')
       if(features_list=='Select all'):
           features_list=['P2P','Mean','STD','RMS','MAV','Energy','Log Energy','Sha Entropy','Low Frequency Gain','Three DB Bandwidth','Unit Gain Frequency','Phase Margin','Gain Margin']
       print(features_list)
       dir_path=os.path.dirname(__file__)
       new_extract_path=dir_path+"//test//"
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
       lfg= []
       tdb=[] 
       ugf=[]
       phm=[]
       gm=[]
       names=[]
       for fi in filelist:   
                name=os.path.basename(fi).split('/')[-1]
                names.append(name)               
                df = pd.read_csv(fi)
                if('dB20(VF(/net_09)/VF(/vdd))' in df.columns):
                    gain_temp = df['dB20(VF(/net_09)/VF(/vdd))']
                    gain_temp=gain_temp/20
                    gain_inv=(10**gain_temp)
                    df['actual_gain']=gain_inv
                    p2pvalue = p2p(df,"actual_gain")     
                    p2pvalues.append(p2pvalue)
                    meancalcvalue = meancalc(df,"actual_gain")
                    meancalcvalues.append(meancalcvalue)       
                    stdcalcvalue = stdcalc(df,"actual_gain")
                    stdcalcvalues.append(stdcalcvalue)
                    rmscalcvalue = rmscalc (df,"actual_gain")
                    rmscalcvalues.append(rmscalcvalue)
                    mavcalcvalue = mavcalc(df,"actual_gain")
                    mavcalcvalues.append(mavcalcvalue)
                    energycalcvalue = energycalc(df,"actual_gain")
                    energycalcvalues.append(energycalcvalue)
                    logenergycalcvalue = log_energycalc(df,"actual_gain")
                    logenergycalcvalues.append(logenergycalcvalue)
                    shaentropy_calcvalue = shaentropy_calc(df,"actual_gain")
                    shaentropy_calcvalues.append(shaentropy_calcvalue)   
                    freq = df.iloc[:,4]
                    freq_new=np.log10(freq)
                    gain = df.iloc[:,3]
                    phase=df.iloc[:,2]
                    freq_arr = freq.to_numpy()
                    gain_arr = gain.to_numpy()
                    for i in range (100):
                        if (freq_arr[i] == 1):
                            gain_value = gain_arr[i]
                    lfg.append(gain_value)
                    f = freq[0+3]
                    tdb.append(f)            
                    #magnitude response graph
                    c, = plt.plot(freq_new,gain)
                    xdata = c.get_xdata()
                    ydata = c.get_ydata()
                    x = 0
                    for i in range(100):
                        if(abs(df['dB20(VF(/net_09)/VF(/vdd))'][i]) >= 0 and abs(df['dB20(VF(/net_09)/VF(/vdd))'][i]) <=1 ):
                            x = i
                        if (x==0):
                            if(abs(df['dB20(VF(/net_09)/VF(/vdd))'][i]) >= 1 and abs(df['dB20(VF(/net_09)/VF(/vdd))'][i]) <2 ):
                                x = i                 
                    d = xdata[x]
                    h = 10**d
                    pm = 180 - abs(phase[x])
                    #unit gain frequency
                    ugf.append(h)
                    phm.append(pm)
                    #gain margin
                    temp=0
                    for i in range(101):
                        if(freq[i] == 1):
                            temp=gain_inv[i]
                    gm.append(temp)    
                else:
                    return render_template('Wrong_input_ac.html')           
       csv_file_path11=dir_path+"//uploads//calculated_values_uploaded_train_ac.csv"
       field_names = ['NAME', 'P2P', 'MEAN', 'MAV', 'STD', 'RMS','ENERGY', 'LOG_ENERGY','SHA_ENTROPY','LOW_FREQUENCY_GAIN','THREE_DB_BANDWIDTH','UNIT_GAIN_FREQUENCY','GAIN_MARGIN','PHASE_MARGIN']
       features = {'NAME':names, 'P2P':p2pvalues, 'MEAN':meancalcvalues, 'MAV':mavcalcvalues, 'STD':stdcalcvalues, 'RMS':rmscalcvalues,'ENERGY':energycalcvalues, 'LOG_ENERGY':logenergycalcvalues,'SHA_ENTROPY':shaentropy_calcvalues,'LOW_FREQUENCY_GAIN':lfg,'THREE_DB_BANDWIDTH':tdb,'UNIT_GAIN_FREQUENCY':ugf,'GAIN_MARGIN':gm,'PHASE_MARGIN':phm}
       with open(csv_file_path11, 'w') as f_object:     
            writer_object = writer(f_object)
            writer_object.writerow(field_names)
            writer_object.writerows(zip(*features.values()))
            f_object.close()
    #Converting the same file to a dataframe
       df=pd.read_csv(csv_file_path11)
       image_path=dir_path+"//static//"
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['P2P'].plot(kind='box')
       title = "P2P"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"P2P_AC.jpg")       
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['MEAN'].plot(kind='box')
       title = "MEAN"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"mean_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['STD'].plot(kind='box')
       title = "STD"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"std_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['RMS'].plot(kind='box')
       title = "RMS"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"rms_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['MAV'].plot(kind='box')
       title = "MAV"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"mav_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['ENERGY'].plot(kind='box')
       title = "ENERGY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"energy_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['LOG_ENERGY'].plot(kind='box')
       title = "LOG_ENERGY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"log energy_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['SHA_ENTROPY'].plot(kind='box')
       title = "SHA_ENTROPY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"sha entropy_AC.jpg")

       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['LOW_FREQUENCY_GAIN'].plot(kind='box')
       title = "LOW_FREQUENCY_GAIN"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"Low Frequency Gain_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['THREE_DB_BANDWIDTH'].plot(kind='box')
       title = "THREE_DB_BANDWIDTH"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"Three DB Bandwidth_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['UNIT_GAIN_FREQUENCY'].plot(kind='box')
       title = "UNIT_GAIN_FREQUENCY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"Unit Gain Frequency_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['PHASE_MARGIN'].plot(kind='box')
       title = "PHASE_MARGIN"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"Phase Margin_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['GAIN_MARGIN'].plot(kind='box')
       title = "GAIN_MARGIN"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"Gain Margin_AC.jpg")

       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       freq = df.iloc[:,4]
       plt.plot(freq,df['GAIN_MARGIN'])
       plt.xlabel('Gain Margin')
       plt.ylabel('Frequency')
       title = "Magnitude Response Graph"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"Magnitude Response Graph_AC.jpg")

       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       freq = df.iloc[:,4]
       plt.plot(freq,df['PHASE_MARGIN'])
       plt.xlabel('Phase Margin')
       plt.ylabel('Frequency')
       title = "Phase Response Graph"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"Phase Response Graph_AC.jpg")
       images=[]
       if(features_list=='Select all'):
           features_list=['P2P','Mean','STD','RMS','MAV','Energy','Log Energy','Sha Entropy','Low Frequency Gain','Three DB Bandwidth','Unit Gain Frequency','Phase Margin','Gain Margin']
       for features in features_list:
           image_name=features+"_AC.jpg"
           images.append(image_name)
       df = pd.read_csv(csv_file_path11)
       stat = df.describe()
       stat.to_csv(dir_path+"//uploads//ac_stat.csv")
       outlier_p2p = find_outliers_IQR_AC(df.iloc[:,[0,1]])
       outlier_p2p.to_csv(dir_path+"//uploads//ac_outliers_P2P.csv")
    
       outlier_offset = find_outliers_IQR_AC(df.iloc[:,[0,3]])
       outlier_offset.to_csv(dir_path+"//uploads//ac_outliers_Mav.csv")
 
       outlier_offset = find_outliers_IQR_AC(df.iloc[:,[0,6]])
       outlier_offset.to_csv(dir_path+"//uploads//ac_outliers_Energy.csv") 
    
       outlier_offset = find_outliers_IQR_AC(df.iloc[:,[0,7]])
       outlier_offset.to_csv(dir_path+"//uploads//ac_outliers_Log Energy.csv")
    
       outlier_offset = find_outliers_IQR_AC(df.iloc[:,[0,8]])
       outlier_offset.to_csv(dir_path+"//uploads//ac_outliers_Sha Entropy.csv")
    
       outlier_offset = find_outliers_IQR_AC(df.iloc[:,[0,5]])
       outlier_offset.to_csv(dir_path+"//uploads//ac_outliers_RMS.csv")

       outlier_offset = find_outliers_IQR_AC(df.iloc[:,[0,2]])
       outlier_offset.to_csv(dir_path+"//uploads//ac_outliers_Mean.csv")
   
       outlier_offset = find_outliers_IQR_AC(df.iloc[:,[0,4]])
       outlier_offset.to_csv(dir_path+"//uploads//ac_outliers_STD.csv")

       outlier_offset = find_outliers_IQR_AC(df.iloc[:,[0,9]])
       outlier_offset.to_csv(dir_path+"//uploads//ac_outliers_Low Frequency Gain.csv")

       outlier_offset = find_outliers_IQR_AC(df.iloc[:,[0,10]])
       outlier_offset.to_csv(dir_path+"//uploads//ac_outliers_Three DB Bandwidth.csv")

       outlier_offset = find_outliers_IQR_AC(df.iloc[:,[0,11]])
       outlier_offset.to_csv(dir_path+"//uploads//ac_outliers_Unit Gain Frequency.csv")

       outlier_offset = find_outliers_IQR_AC(df.iloc[:,[0,12]])
       outlier_offset.to_csv(dir_path+"//uploads//ac_outliers_Gain Margin.csv")

       outlier_offset = find_outliers_IQR_AC(df.iloc[:,[0,13]])
       outlier_offset.to_csv(dir_path+"//uploads//ac_outliers_Phase Margin.csv")
       list_df=[]
       if(features_list=='Select all'):
           features_list=['P2P','Mean','STD','RMS','MAV','Energy','Log Energy','Sha Entropy','Low Frequency Gain','Three DB Bandwidth','Unit Gain Frequency','Phase Margin','Gain Margin']
       for features in features_list:
           outlier_file_name=dir_path+"//uploads//ac_outliers_"+features+".csv"
           
           df=pd.read_csv(outlier_file_name)
           
           if(df.empty==False):
               df1=df.iloc[:,[1,2]]
               list_df.append(df1)
               
       return render_template('box_plots_display_ac.html', images=images,list_df=list_df)
@app.route('/clustering_ac',methods=['GET','POST'])  
def clustering_ac():  
    return render_template("clustering_main_page_ac.html")  
@app.route('/app2_clustering_ac',methods=['GET','POST'])
def app2_clustering_ac():
    df=pd.read_csv(csv_file_path11)
    scaler=MinMaxScaler()
    columns=df.columns.to_list()
    for_scaling=columns[1:14]
    df_fn=df[for_scaling]
    scaled_values=scaler.fit_transform(df_fn)
    scaled_df=pd.DataFrame(scaled_values,columns=[for_scaling])
    scaled_df['Name']=df['NAME']
    csv_file_path12 = dir_path+"//uploads//ac_parameter_scaled.csv"
    scaled_df.to_csv(csv_file_path12)
    if request.method == "POST":
        value = request.form.get('Clustering')
    if value=='1':
        return redirect(url_for('app3_KMeansclustering_ac'))
    elif value=='2':
        return redirect(url_for('app4_OneClassSVMClustering_ac'))
    elif value=='3':
        return redirect(url_for('app5_DBScanClustering_ac'))
@app.route('/app3_KMeansclustering_ac',methods=['GET','POST'])
def app3_KMeansclustering_ac():
    return render_template("KMeans_Clustering_ac.html") 
@app.route('/app3_logic_ac',methods=['GET','POST'])
def app3_logic_ac():
    if request.method == "POST":
        num_clusters=request.form.get("Cluster Number")
        num_clusters=int(num_clusters)
        df=pd.read_csv(csv_file_path11)
        X = df.iloc[:,1:14].values
        kmeans = KMeans(num_clusters)
        kmeans.fit(X)
        identified_clusters = kmeans.fit_predict(X)
        df['Clusters'] = identified_clusters 
        #To store the clusters in the file
        print(pickle_path_kmeans_ac)
        pickle.dump(kmeans, open(pickle_path_kmeans_ac,'wb'))
        df.to_csv(csv_file_path13)
    return render_template("KMeans_Output_ac.html")
@app.route('/app4_OneClassSVMClustering_ac',methods=['GET','POST'])
def app4_OneClassSVMClustering_ac():
    return render_template('OneClassSVM_ac.html')
@app.route('/app4_logic_ac',methods=['GET','POST'])
def app4_logic_ac():
    if request.method == "POST":
        gamma_num=request.form.get("Gamma")
        gamma_num=float(gamma_num)
        Nu_num=request.form.get("Nu")
        Nu_num=float(Nu_num)
    df=pd.read_csv(csv_file_path11)
    columns=df.columns.to_list()
    columns=columns[1:14]
    df_svm=df[columns].values
    #OneClassSVM using rbf (radial basis function)
    clf=OneClassSVM(kernel='rbf',gamma=gamma_num,nu=Nu_num).fit(df_svm)
    clusters=clf.predict(df_svm)
    for_scaling=columns[1:]
    df=df[for_scaling]
    df['cluster']=clusters
    df.to_csv(csv_file_path14)
    pickle.dump(clf, open(pickle_path_svm_ac,'wb'))
    #PCA Plot
    df=pd.read_csv(csv_file_path11)
    sc = StandardScaler()
    scaleddata = sc.fit_transform(df.iloc[:,1:14])
    pca = PCA(n_components = 3)
    dataset_pca = pca.fit_transform(scaleddata)
    df_pca = pd.DataFrame(dataset_pca,columns=['PCA1','PCA2','PCA3'])
    df_pca.to_csv(csv_file_path15)
    data=[]
    for d in df_pca.itertuples():
        my_list =[d.PCA1, d.PCA2, d.PCA3]
        data.extend([my_list])
    clf=OneClassSVM(kernel='rbf',gamma=3,nu=0.001).fit(data)
    clusters=clf.predict(data)
    df_pca['OneClassSVM']=clusters
    pickle.dump(clf, open(pickle_path_svm_pca_ac,'wb'))
    clusters = DBSCAN(eps=1.0, min_samples=15000).fit(data)
    labels=clusters.labels_
    df_pca['DBSCAN'] = clusters.labels_
    pickle.dump(clusters,open(pickle_path_dbscan_pca_ac,'wb'))  
    df_pca.to_csv(csv_file_path16)
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes(projection='3d')
    df_pca_svm=df_pca.iloc[:,0:4]
    df_pca_dscan=df_pca.iloc[:,[col for col in range(len(df_pca.columns)) if col != 3]]
    for i in df_pca_svm:
           xdata = df_pca_svm['PCA1']
           zdata = df_pca_svm['PCA2']
           ydata = df_pca_svm['PCA3']
           ax.scatter3D(xdata, ydata, zdata)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.legend()
    plt.title("3D PCA plot for SVM")
    image_path=dir_path+"//static//"
    plt.savefig(image_path+"PCA_SVM_ac.jpg")
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes(projection='3d')
    for i in df_pca_dscan:
           xdata = df_pca_dscan['PCA1']
           zdata = df_pca_dscan['PCA2']
           ydata = df_pca_dscan['PCA3']
           ax.scatter3D(xdata, ydata, zdata)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.legend()
    plt.title("3D PCA plot for DBSCAN")
    plt.savefig(image_path+"PCA_DBSCAN_ac.jpg")
    return render_template("SVM_Output_ac.html",image="PCA_SVM_ac.jpg")
@app.route('/app5_DBScanClustering_ac',methods=['GET','POST'])
def app5_DBScanClustering_ac():
    return render_template('DBScanClustering_ac.html')
@app.route('/app5_logic_ac',methods=['GET','POST'])
def app5_logic_ac():
    if request.method == "POST":
        epsilon_num=request.form.get("epsi")
        epsilon_num=float(epsilon_num)
    df = pd.read_csv(csv_file_path12)
    df_fn= df.iloc[:,1:14].values
    clusters = DBSCAN(eps=epsilon_num, min_samples=1000).fit(df_fn)
    labels=clusters.labels_
    df['cluster'] = clusters.labels_  
    pickle.dump(clusters,open(pickle_path_dbscan_ac,'wb'))
    df.to_csv(csv_file_path20)
    return render_template("DBScan_Output_ac.html",image="PCA_DBSCAN_ac.jpg")   
@app.route('/testing_page_ac',methods=['GET','POST'])  
def testing_page_ac():  
    return render_template("input_test_file_ac.html")  
@app.route('/app6_testing_ac',methods=['GET','POST'])
def app6_testing_ac():
   if request.method == "POST":
       f = request.files['file']
       f.save(f.filename)       
       features_list=request.form.getlist('Features')
       if(features_list=='Select all'):
           features_list=['P2P','Mean','STD','RMS','MAV','Energy','Log Energy','Sha Entropy','Low Frequency Gain','Three DB Bandwidth','Unit Gain Frequency','Phase Margin','Gain Margin']
       dir_path=os.path.dirname(__file__)
       df = pd.read_csv(f.filename)
       name=f.filename
       if('dB20(VF(/net_09)/VF(/vdd))' in df.columns):
            gain_temp = df['dB20(VF(/net_09)/VF(/vdd))']
            gain_temp=gain_temp/20
            gain_inv=(10**gain_temp)
            df['actual_gain']=gain_inv
            p2pvalue = p2p(df,"actual_gain")     
            meancalcvalue = meancalc(df,"actual_gain") 
            stdcalcvalue = stdcalc(df,"actual_gain")
            rmscalcvalue = rmscalc (df,"actual_gain")         
            mavcalcvalue = mavcalc(df,"actual_gain")       
            energycalcvalue = energycalc(df,"actual_gain")     
            logenergycalcvalue = log_energycalc(df,"actual_gain")       
            shaentropy_calcvalue = shaentropy_calc(df,"actual_gain")         
            freq = df.iloc[:,4]
            freq_new=np.log10(freq)
            gain = df.iloc[:,3]
            phase=df.iloc[:,2]
            freq_arr = freq.to_numpy()
            gain_arr = gain.to_numpy()
            for i in range (100):
                        if (freq_arr[i] == 1):
                            gain_value = gain_arr[i]    
            f = freq[0+3]
            c, = plt.plot(freq_new,gain)
            xdata = c.get_xdata()
            ydata = c.get_ydata()
            x = 0
            for i in range(100):
                        if(abs(df['dB20(VF(/net_09)/VF(/vdd))'][i]) >= 0 and abs(df['dB20(VF(/net_09)/VF(/vdd))'][i]) <=1 ):
                            x = i
                        if (x==0):
                            if(abs(df['dB20(VF(/net_09)/VF(/vdd))'][i]) >= 1 and abs(df['dB20(VF(/net_09)/VF(/vdd))'][i]) <2 ):
                                x = i       
            d = xdata[x]
            h = 10**d
            pm = 180 - abs(phase[x])                    
            temp=0
            for i in range(101):
                        if(freq[i] == 1):
                            temp=gain_inv[i]
       else:
            return render_template('Wrong_Input_ac.html')          
       field_names = ['NAME', 'P2P', 'MEAN', 'MAV', 'STD', 'RMS','ENERGY', 'LOG_ENERGY','SHA_ENTROPY','LOW_FREQUENCY_GAIN','THREE_DB_BANDWIDTH','UNIT_GAIN_FREQUENCY','GAIN_MARGIN','PHASE_MARGIN']
       features = {'NAME':name, 'P2P':p2pvalue, 'MEAN':meancalcvalue, 'MAV':mavcalcvalue, 'STD':stdcalcvalue, 'RMS':rmscalcvalue,'ENERGY':energycalcvalue, 'LOG_ENERGY':logenergycalcvalue,'SHA_ENTROPY':shaentropy_calcvalue,'LOW_FREQUENCY_GAIN':gain_value,'THREE_DB_BANDWIDTH':f,'UNIT_GAIN_FREQUENCY':h,'GAIN_MARGIN':temp,'PHASE_MARGIN':pm}
    #writer() function returns a writer object that converts the user's data into a delimited string. This string can later be used to write into CSV files using the writerow() function
       with open(csv_file_path19, 'w') as f_object:     
            writer_object = writer(f_object)
            writer_object.writerow(field_names)
            writer_object.writerow(features.values())
            f_object.close()
    #Converting the same file to a dataframe
       df=pd.read_csv(csv_file_path19)
       image_path=dir_path+"//static//"
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['P2P'].plot(kind='box')
       title = "P2P"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"P2P_test_ac.jpg")      
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['MEAN'].plot(kind='box')
       title = "MEAN"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"mean_test_ac.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['STD'].plot(kind='box')
       title = "STD"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"std_test_ac.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['RMS'].plot(kind='box')
       title = "RMS"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"rms_test_ac.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['MAV'].plot(kind='box')
       title = "MAV"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"mav_test_ac.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['ENERGY'].plot(kind='box')
       title = "ENERGY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"energy_test_ac.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['LOG_ENERGY'].plot(kind='box')
       title = "LOG_ENERGY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"log energy_test_ac.jpg")

       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['SHA_ENTROPY'].plot(kind='box')
       title = "SHA_ENTROPY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"sha entropy_test_ac.jpg")

       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['LOW_FREQUENCY_GAIN'].plot(kind='box')
       title = "LOW_FREQUENCY_GAIN"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"Low Frequency Gain_test_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['THREE_DB_BANDWIDTH'].plot(kind='box')
       title = "THREE_DB_BANDWIDTH"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"Three DB Bandwidth_test_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['UNIT_GAIN_FREQUENCY'].plot(kind='box')
       title = "UNIT_GAIN_FREQUENCY"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"Unit Gain Frequency_test_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['PHASE_MARGIN'].plot(kind='box')
       title = "PHASE_MARGIN"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"Phase Margin_test_AC.jpg")
    
       fig=plt.figure(figsize=(5,5),facecolor='w', edgecolor='k')
       df['GAIN_MARGIN'].plot(kind='box')
       title = "GAIN_MARGIN"
       plt.title(title,fontsize='25')
       plt.savefig(image_path+"Gain Margin_test_AC.jpg")      
       images=[]
       if(features_list=='Select all'):
                features_list=['P2P','Mean','STD','RMS','MAV','Energy','Log Energy','Sha Entropy','Low Frequency Gain','Three DB Bandwidth','Unit Gain Frequency','Phase Margin','Gain Margin']
       for features in features_list:
           
           image_name=features+"_test_AC.jpg"
           images.append(image_name)
           #Converting the same file to a dataframe
       df=pd.read_csv(csv_file_path19)
       data = df.iloc[:,1:14].values
    #Reading the pickle file for the Kmeans
       kmodel = pickle.load(open(pickle_path_kmeans_ac,'rb'))
       k = kmodel.predict(data)
       k = str(k)
    #Reading the pickle file for the SVM
       smodel = pickle.load(open(pickle_path_svm_ac,'rb'))
       s = smodel.predict(data)
       s = str(s)
    #Reading the pickle file for the dbscan
       dmodel=pickle.load(open('dmodel_ac.pkl','rb'))
       d = dmodel.fit_predict(data, y=None, sample_weight=None)
       d = str(d)     
       return render_template('Cluster_test_ac.html', images=images,k=k,d=d,s=s) 
@app.route('/app7_ac',methods=['GET','POST'])
def app7_ac():
       df=pd.read_csv(csv_file_path11)
       df1=pd.read_csv(csv_file_path19)
       data = df1.iloc[:,0:14].values
       list1=data[0]
       df.loc[len(df)] = list1
       sc = StandardScaler()
       scaleddata = sc.fit_transform(df.iloc[:,1:14])
       pca = PCA(n_components = 3)
       dataset_pca = pca.fit_transform(scaleddata)
       df_pca = pd.DataFrame(dataset_pca,columns=['PCA1','PCA2','PCA3'])
       df_pca.to_csv(csv_file_path17)
       data1=[]
       for d in df_pca.itertuples():
        my_list =[d.PCA1, d.PCA2, d.PCA3]
        data1.extend([my_list])
       clf=OneClassSVM(kernel='rbf',gamma=3,nu=0.001).fit(data1)
       clusters=clf.predict(data1)
       df_pca['OneClassSVM']=clusters
       clusters = DBSCAN(eps=1.0, min_samples=15000).fit(data1)
       labels=clusters.labels_
       df_pca['DBSCAN'] = clusters.labels_
       df_pca.to_csv(csv_file_path18)
       fig = plt.figure(figsize=(7,7))
       ax = plt.axes(projection='3d')
       df_pca_svm=df_pca.iloc[:,0:4]
       df_pca_dscan=df_pca.iloc[:,[col for col in range(len(df_pca.columns)) if col != 3]]
       for i in df_pca_svm:
           xdata = df_pca_svm['PCA1']
           zdata = df_pca_svm['PCA2']
           ydata = df_pca_svm['PCA3']
           ax.scatter3D(xdata, ydata, zdata)
       ax.set_xlabel('PCA1')
       ax.set_ylabel('PCA2')
       ax.set_zlabel('PCA3')
       ax.legend()
       plt.title("3D PCA plot for SVM")
       image_path=dir_path+"//static//"
       plt.savefig(image_path+"PCA_SVM_test_ac.jpg")
       fig = plt.figure(figsize=(7,7))
       ax = plt.axes(projection='3d')
       for i in df_pca_dscan:
           xdata = df_pca_dscan['PCA1']
           zdata = df_pca_dscan['PCA2']
           ydata = df_pca_dscan['PCA3']
           ax.scatter3D(xdata, ydata, zdata)
       ax.set_xlabel('PCA1')
       ax.set_ylabel('PCA2')
       ax.set_zlabel('PCA3')
       ax.legend()
       plt.title("3D PCA plot for DBSCAN")
       plt.savefig(image_path+"PCA_DBSCAN_test_ac.jpg")
       image1= 'PCA_DBSCAN_Test_ac.jpg'
       image2='PCA_SVM_Test_ac.jpg'
       return render_template('PCATest_ac.html',image1=image1,image2=image2)   
if __name__ == '__main__':
    app.run(debug=True)