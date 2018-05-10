from sklearn.decomposition import PCA   
pca=PCA(n_components=2)  

newData=pca.fit_transform(data)  

with open("Output_pca.txt", "w") as text_file:
   for item in newData:
   		text_file.write("%s,\n" % item)