from flask import *
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
app = Flask(__name__)
dir_path=os.path.dirname(__file__)
csv_file_path=dir_path+"\\uploads\\calculated_values_uploaded.csv"
csv_file_path1=dir_path+"\\uploads\\opamp_parameter_scaled.csv"
csv_file_path2=dir_path+"\\uploads\\kmeans_opamp.csv"
csv_file_path3=dir_path+"\\uploads\\svm_opamp.csv"
csv_file_path4=dir_path+"\\uploads\\dbscan_opamp.csv"
csv_file_path5=dir_path+"\\uploads\\opamp_pca.csv"
csv_file_path6=dir_path+"\\uploads\\opamp_pca_clustering"
pickle_path_kmeans=dir_path+"\\kmodel_opamp.pkl"
pickle_path_svm=dir_path+"\\smodel_opamp.pkl"
pickle_path_dbscan=dir_path+"\\dmodel_opamp.pkl"
pickle_path_svm_pca=dir_path+"\\smodel_opamp_pca.pkl"
pickle_path_dbscan_pca=dir_path+"\\dmodel_opamp_pca.pkl"
@app.route('/',methods=['GET','POST'])  
def main():  
    return render_template("clustering_main_page.html")  
@app.route('/app2_clustering',methods=['GET','POST'])

def app2_clustering():
    df=pd.read_csv(csv_file_path)
    scaler=MinMaxScaler()
    columns=df.columns.to_list()
    for_scaling=columns[1:10]
    df_fn=df[for_scaling]
    scaled_values=scaler.fit_transform(df_fn)
    scaled_df=pd.DataFrame(scaled_values,columns=[for_scaling])
    scaled_df['Name']=df['NAME']
    csv_file_path1 = dir_path+"\\uploads\\opamp_parameter_scaled.csv"
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
# def kmeans_page(): 
#     return render_template("KMeans_Clustering.html") 
def app3_KMeansclustering():
    return render_template("KMeans_Clustering.html") 
@app.route('/app3_logic',methods=['GET','POST'])
def app3_logic():
    if request.method == "POST":
        num_clusters=request.form.get("Cluster Number")
        num_clusters=int(num_clusters)
        df=pd.read_csv(csv_file_path1)
        X = df.iloc[:,1:9].values
        #Clustering with two clusters
        kmeans = KMeans(num_clusters)
        kmeans.fit(X)
        identified_clusters = kmeans.fit_predict(X)
        df['Clusters'] = identified_clusters 
        
        #To store the clusters in the file
        pickle.dump(kmeans, open(pickle_path_kmeans,'wb'))
        df.to_csv(csv_file_path2)
        

    # Send the CSV file as a download response
    return render_template("KMeans_Output.html")
@app.route('/app4_OneClassSVMClustering',methods=['GET','POST'])

def app4_OneClassSVMClustering():
    df=pd.read_csv(csv_file_path1)
    columns=df.columns.to_list()
    columns=columns[1:9]
    df_svm=df[columns].values
    #OneClassSVM using rbf (radial basis function)
    clf=OneClassSVM(kernel='rbf',gamma=0.1,nu=0.001).fit(df_svm)
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
    image_path=dir_path+"\\static\\"
    plt.savefig(image_path+"PCA_SVM.jpg")
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
    plt.savefig(image_path+"PCA_DBSCAN.jpg")
    return render_template("SVM_Output.html",image="PCA_SVM.jpg")
@app.route('/app5_DBScanClustering',methods=['GET','POST'])
def app5_DBScanClustering():
    df = pd.read_csv(csv_file_path1)
    df_fn= df.iloc[:,1:9].values
    clusters = DBSCAN(eps=0.7, min_samples=1000).fit(df_fn)
    labels=clusters.labels_
    df['cluster'] = clusters.labels_
    
    pickle.dump(clusters,open(pickle_path_dbscan,'wb'))
    df.to_csv(csv_file_path4)
    return render_template("DBScan_Output.html",image="PCA_DBSCAN.jpg")

if __name__ == '__main__':
    app.run(debug=True)