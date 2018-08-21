# Projeto 6: Classic Iris Data Set.

Este é talvez o banco de dados mais conhecido encontrado na literatura sobre reconhecimento de padrões. O papel de Fisher é um clássico
no campo e é referenciado com frequência até hoje. (Veja Duda & Hart, por exemplo.) O conjunto de dados contém 3 classes de 50 instâncias
cada, onde cada classe se refere a um tipo de planta da íris. Uma classe é linearmente separável das outras duas; os últimos 
NÃO são linearmente separáveis um do outro.

Atributo previsto: classe da planta da íris.

Este é um domínio extremamente simples.
Estes dados diferem dos dados apresentados no artigo de Fishers (identificado por Steve Chadwick, spchadwick@espeedaz.net). 
A 35ª amostra deve ser: 4.9.3.1.1.5,0.2, "Iris-setosa", onde o erro está no quarto recurso. A 38ª amostra: 4.9.3.6.1.4.0.1, "Iris-setosa", onde os erros estão na segunda e terceira características.

## Voting Classifier

### Import and input data

```{python, cache=FALSE, message=FALSE, warning=FALSE}
from sklearn import datasets
Iris = datasets.load_iris()
X = Iris.data[:, [2, 3]]
y = Iris.target

color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA']#, '#AAAAFF']
color_list_bold = ['#EEEE00', '#000000', '#00CC00']#, '#0000CC']

custom_cmap2 = ListedColormap(color_list_light)
custom_cmap1 = ListedColormap(color_list_bold)
```

### Training classifiers
```{python, cache=FALSE, message=FALSE, warning=FALSE}
clf1 = DecisionTreeClassifier(max_depth=5)
clf2 = KNeighborsClassifier(n_neighbors=6)
clf3 = SVC(kernel='rbf', probability=True,gamma=5,C=1)
eclf = VotingClassifier(estimators=[('dt', clf1),('knn', clf2),('svc', clf3)],voting='soft',weights=[2, 1, 2])

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)
```

### Plotting decision regions
```{python, cache=FALSE, message=FALSE, warning=FALSE}
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))
for idx, clf, tt in zip(product([0, 1], [0, 1]), [clf1, clf2, clf3, eclf], 
                       ['Decision Tree (depth=5)', 'KNN (k=6)','Kernel SVM', 'Soft Voting']):
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.8,cmap=custom_cmap2)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,cmap=custom_cmap1,s=20, edgecolor='black')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()
```
